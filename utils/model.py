from .fbank import ReadCMVN
from . import tokens
import datetime as dt
import logging, torch

from icefall.utils import AttributeDict
from icefall.hifigan.config import v1,v2,v3
from icefall.hifigan.models import Generator as HiFiGAN
from icefall.hifigan.denoiser import Denoiser
from icefall.tokenizer import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

import sys, pathlib
from icefall.models.matcha_tts import MatchaTTS

import platform
IsWindows = platform.system() == 'Windows'


class Model:
	def __init__(self, tokenizer: Tokenizer, model: MatchaTTS, confs, n_timesteps: int = 2, length_scale: float = 1.0, temperature: float = 0.667) -> None:
		self.tokenizer = tokenizer
		self.model = model
		self.n_timesteps = n_timesteps
		self.length_scale = length_scale
		self.temperature = temperature
		self.device = 'cpu'
		self.params = AttributeDict({
            "model_args": confs,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": -1,  # 0
            "log_interval": 10,
            "valid_interval": 1500,
        })

	def GetParams(self):
		return self.params

	def GetModel(self) -> MatchaTTS:
		return self.model
	
	def UploadToDevice(self, device):
		self.model.to(device)
		self.device = device
	
	def LoadCheckpoint(self, filename: str):
		# Ref: https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
		if IsWindows:
			temp = pathlib.PosixPath
			pathlib.PosixPath = pathlib.WindowsPath

		logging.info(f"Loading checkpoint from {filename}")
		checkpoint = torch.load(filename, map_location="cpu")
		strict = False
		if next(iter(checkpoint["model"])).startswith("module."):
			logging.info("Loading checkpoint saved by DDP")

			dst_state_dict = self.model.state_dict()
			src_state_dict = checkpoint["model"]
			for key in dst_state_dict.keys():
				src_key = "{}.{}".format("module", key)
				dst_state_dict[key] = src_state_dict.pop(src_key)
			assert len(src_state_dict) == 0
			self.model.load_state_dict(dst_state_dict, strict=strict)
		else:
			self.model.load_state_dict(checkpoint["model"], strict=strict)

		checkpoint.pop("model")

		if IsWindows:
			pathlib.PosixPath = temp
		return checkpoint
	
	def Encode(self, text:str):
		text_processed = tokens.process_text(text=text, tokenizer=self.tokenizer, device=self.device)
		return {
			"text": text,
			"x": text_processed["x"],
			"x_lengths": text_processed["x_lengths"],
		}

	def Forward(self, encoded: dict):
		if isinstance(self.model, DDP):
			return self.model.module.synthesise(
				encoded["x"],
				encoded["x_lengths"],
				n_timesteps=self.n_timesteps,
				temperature=self.temperature,
				spks=None,
				length_scale=self.length_scale,
			)
		else:
			return self.model.synthesise(
				encoded["x"],
				encoded["x_lengths"],
				n_timesteps=self.n_timesteps,
				temperature=self.temperature,
				spks=None,
				length_scale=self.length_scale,
			)

	def __call__(self, text: str):
		text_processed = tokens.process_text(text=text, tokenizer=self.tokenizer, device=self.device)
		start_t = dt.datetime.now()
		output = self.model.synthesise(
			text_processed["x"],
			text_processed["x_lengths"],
			n_timesteps=self.n_timesteps,
			temperature=self.temperature,
			spks=None,
			length_scale=self.length_scale,
		)
		# merge everything to one dict
		output.update({"start_t": start_t, **text_processed})
		return output


class ModelBuilder:
	def __init__(self) -> None:
		self.n_feats = 80
		self.filter_channels_dp = 256
		self.encoder_params_p_dropout = 0.1
		self.mel_mean = 0
		self.mel_std = 1
		self.n_timesteps = 2
		self.length_scale = 1.0

	def LoadTokenizer(self, token_filename: str):
		self.tokenizer = Tokenizer(token_filename)

	def LoadCMVN(self, cmvn_filename: str):
		cmvn = ReadCMVN(cmvn_filename)
		self.mel_mean, self.mel_std = cmvn["fbank_mean"], cmvn["fbank_std"]
		if "sampling_rate" in cmvn:
			self.sampling_rate = cmvn["sampling_rate"]
		else:
			self.sampling_rate = None

	def GetSamplingRate(self):
		"""
		call LoadCMVN() first
		"""
		return self.sampling_rate

	def BuildModel(self):
		params = AttributeDict({
			"n_spks": 1,  # for baker-zh.
			"spk_emb_dim": 64,
			"n_feats": self.n_feats,
			"out_size": None,  # or use 172
			"prior_loss": True,
			"use_precomputed_durations": False,
			"n_vocab": self.tokenizer.vocab_size,
			"data_statistics": AttributeDict({
				"mel_mean": self.mel_mean,
				"mel_std": self.mel_std,
			}),
			"encoder": AttributeDict({
				"encoder_type": "RoPE Encoder",  # not used
				"encoder_params": AttributeDict({
					"n_feats": self.n_feats,
					"n_channels": 192,
					"filter_channels": 768,
					"filter_channels_dp": self.filter_channels_dp,
					"n_heads": 2,
					"n_layers": 6,
					"kernel_size": 3,
					"p_dropout": self.encoder_params_p_dropout,
					"spk_emb_dim": 64,
					"n_spks": 1,
					"prenet": True,
				}),
				"duration_predictor_params": AttributeDict({
					"filter_channels_dp": self.filter_channels_dp,
					"kernel_size": 3,
					"p_dropout": self.encoder_params_p_dropout,
				}),
			}),
			"decoder": AttributeDict({
				"channels": [256, 256],
				"dropout": 0.05,
				"attention_head_dim": 64,
				"n_blocks": 1,
				"num_mid_blocks": 2,
				"num_heads": 2,
				"act_fn": "snakebeta",
			}),
			"cfm": AttributeDict({
				"name": "CFM",
				"solver": "euler",
				"sigma_min": 1e-4,
			}),
			"optimizer": AttributeDict({
				"lr": 1e-4,
				"weight_decay": 0.0,
			}),
		})

		model = Model(
			self.tokenizer,
			MatchaTTS(**params),
			params
		)
		return model
	
class MelDecoder:
	def __init__(self, vocoder_filename: str, device="cpu"):
		self.vocoder_filename = vocoder_filename

		self.load_vocoder_and_denoiser(device=device)

	def load_vocoder_and_denoiser(self, device="cpu"):
		if self.vocoder_filename.endswith("v1"):
			h = AttributeDict(v1)
		elif self.vocoder_filename.endswith("v2"):
			h = AttributeDict(v2)
		elif self.vocoder_filename.endswith("v3"):
			h = AttributeDict(v3)
		else:
			raise ValueError(f"supports only v1, v2, and v3, given {self.vocoder_filename}")

		hifigan = HiFiGAN(h).to("cpu")
		hifigan.load_state_dict(
			torch.load(self.vocoder_filename, map_location="cpu")["generator"]
		)
		_ = hifigan.eval()
		hifigan.remove_weight_norm()
		self.vocoder = hifigan
		self.vocoder.to(device)
		self.denoiser = Denoiser(self.vocoder, mode="zeros")
		self.denoiser.to(device)

	def UploadToDevice(self, device):
		self.vocoder.to(device)
		self.denoiser.to(device)

	def __call__(self, mel: torch.Tensor):
		"""
		return waveform
		"""
		audio = self.vocoder(mel).clamp(-1, 1)
		audio = self.denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
		return audio.squeeze()
