#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse
import json
import logging
import os
import typing
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Union
import itertools
import heapq

# import before everything
import piper_phonemize

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.utils import fix_random_seed
from icefall.model import fix_len_compatibility
from icefall.models.matcha_tts import MatchaTTS
from icefall.tokenizer import Tokenizer
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from icefall.tts_datamodule import BakerZhTtsDataModule
from icefall.utils import MetricsTracker

from icefall.checkpoint import load_checkpoint, save_checkpoint
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import AttributeDict, setup_logger, str2bool
import torch.utils.data
from utils.model import Model as ModelWrapper, MelDecoder
import pathlib
import platform

IsWindows = platform.system() == 'Windows'

def get_parser():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument(
		"--world-size",
		type=int,
		default=1,
		help="Number of GPUs for DDP training.",
	)

	parser.add_argument(
		"--master-port",
		type=int,
		default=12335,
		help="Master port to use for DDP training.",
	)

	parser.add_argument(
		"--tensorboard",
		type=str2bool,
		default=True,
		help="Should various information be logged in tensorboard.",
	)

	parser.add_argument(
		"--num-epochs",
		type=int,
		default=1000,
		help="Number of epochs to train.",
	)

	parser.add_argument(
		"--start-epoch",
		type=int,
		default=1,
		help="""Resume training from this epoch. It should be positive.
		If larger than 1, it will load checkpoint from
		exp-dir/epoch-{start_epoch-1}.pt
		""",
	)

	parser.add_argument(
		"--exp-dir",
		type=Path,
		default="matcha/exp",
		help="""The experiment dir.
		It specifies the directory where all training related
		files, e.g., checkpoints, log, etc, are saved
		""",
	)

	parser.add_argument(
		"--tokens",
		type=str,
		default="data/tokens.txt",
		help="""Path to vocabulary.""",
	)

	parser.add_argument(
		"--cmvn",
		type=str,
		default="data/fbank/cmvn.json",
		help="""Path to vocabulary.""",
	)

	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="The seed for random generators intended for reproducibility",
	)

	parser.add_argument(
		"--save-every-n",
		type=int,
		default=10,
		help="""Save checkpoint after processing this number of epochs"
		periodically. We save checkpoint to exp-dir/ whenever
		params.cur_epoch % save_every_n == 0. The checkpoint filename
		has the form: f'exp-dir/epoch-{params.cur_epoch}.pt'.
		Since it will take around 1000 epochs, we suggest using a large
		save_every_n to save disk space.
		""",
	)

	parser.add_argument(
		"--use-fp16",
		type=str2bool,
		default=False,
		help="Whether to use half precision training.",
	)

	parser.add_argument(
		"--keep-nepochs",
		type=int,
		default=2,
		help="Keep the number of epochs checkpoints in disk."
	)
	parser.add_argument(
		"--keep-nbest-train",
		type=int,
		default=5,
		help="Keep the number of epochs checkpoints in disk."
	)
	parser.add_argument(
		"--keep-nbest-valid",
		type=int,
		default=5,
		help="Keep the number of epochs checkpoints in disk."
	)
	parser.add_argument(
		"--pretrained-checkpoint",
		type=str,
		default=None,
		help="""Path to the checkpoint to initialize from.""",
	)
	parser.add_argument(
		"--vocoder-checkpoint",
		type=str,
		default=None,
		help="Log audio in validation set during training."
	)
	parser.add_argument(
		"--learning_type",
		type=str,
		default="adamw",
		help="[adamw, adam]"
	)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=1e-3,
		help="Learning rate for Adam optimizer.",
	)
	parser.add_argument(
		"--learning_weight_decay",
		type=float,
		default=0.01,
	)
	parser.add_argument(
		"--log-n-audio",
		type=int,
		default=2,
		help="Log audio in validation set during training."
	)
	parser.add_argument( # ~ 20 steps per epoch 
		"--scheduler_cos_T0",
		type=int,
		default=1000,
	)
	parser.add_argument(
		"--scheduler_cos_mult",
		type=int,
		default=2,
		help="multiply t0 to get next t"
	)
	parser.add_argument(
		"--scheduler_eta_min",
		type=float,
		default=1e-5,
		help="min of scheduler"
	)
	parser.add_argument(
		"--scheduler_auto_peak",
		type=int, # 3
		default=None,
		help="calculate scheduler_cos_T0 by this peak automatically. Notice, this will ignore scheduler_cos_T0"
	)

	return parser


def get_data_statistics():
	return AttributeDict(
		{
			"mel_mean": 0,
			"mel_std": 1,
		}
	)


def _get_data_params() -> AttributeDict:
	params = AttributeDict(
		{
			"name": "baker-zh",
			"train_filelist_path": "./filelists/ljs_audio_text_train_filelist.txt",
			"valid_filelist_path": "./filelists/ljs_audio_text_val_filelist.txt",
			#  "batch_size": 64,
			#  "num_workers": 1,
			#  "pin_memory": False,
			"cleaners": ["english_cleaners2"],
			"add_blank": True,
			"n_spks": 1,
			"n_fft": 1024,
			"n_feats": 80,
			"sampling_rate": 22050,
			"hop_length": 256,
			"win_length": 1024,
			"f_min": 0,
			"f_max": 8000,
			"seed": 1234,
			"load_durations": False,
			"data_statistics": get_data_statistics(),
		}
	)
	return params


def _get_model_params() -> AttributeDict:
	n_feats = 80
	filter_channels_dp = 256
	encoder_params_p_dropout = 0.1
	params = AttributeDict(
		{
			"n_spks": 1,  # for baker-zh.
			"spk_emb_dim": 64,
			"n_feats": n_feats,
			"out_size": None,  # or use 172
			"prior_loss": True,
			"use_precomputed_durations": False,
			"data_statistics": get_data_statistics(),
			"encoder": AttributeDict(
				{
					"encoder_type": "RoPE Encoder",  # not used
					"encoder_params": AttributeDict(
						{
							"n_feats": n_feats,
							"n_channels": 192,
							"filter_channels": 768,
							"filter_channels_dp": filter_channels_dp,
							"n_heads": 2,
							"n_layers": 6,
							"kernel_size": 3,
							"p_dropout": encoder_params_p_dropout,
							"spk_emb_dim": 64,
							"n_spks": 1,
							"prenet": True,
						}
					),
					"duration_predictor_params": AttributeDict(
						{
							"filter_channels_dp": filter_channels_dp,
							"kernel_size": 3,
							"p_dropout": encoder_params_p_dropout,
						}
					),
				}
			),
			"decoder": AttributeDict(
				{
					"channels": [256, 256],
					"dropout": 0.05,
					"attention_head_dim": 64,
					"n_blocks": 1,
					"num_mid_blocks": 2,
					"num_heads": 2,
					"act_fn": "snakebeta",
				}
			),
			"cfm": AttributeDict(
				{
					"name": "CFM",
					"solver": "euler",
					"sigma_min": 1e-4,
				}
			),
			"optimizer": AttributeDict(
				{
					"lr": 1e-4,
					"weight_decay": 0.0,
				}
			),
		}
	)

	return params


def get_params():
	params = AttributeDict(
		{
			"model_args": _get_model_params(),
			"data_args": _get_data_params(),
			"best_train_loss": float("inf"),
			"best_valid_loss": float("inf"),
			"best_train_epoch": -1,
			"best_valid_epoch": -1,
			"batch_idx_train": -1,  # 0
			"log_interval": 10,
			"valid_interval": 1500,
			"env_info": get_env_info(),
		}
	)
	return params


def get_model(params):
	m = MatchaTTS(**params.model_args)
	return m


def load_checkpoint_if_available(
	params: AttributeDict, model: nn.Module
) -> Optional[Dict[str, Any]]:
	"""Load checkpoint from file.

	If params.start_epoch is larger than 1, it will load the checkpoint from
	`params.start_epoch - 1`.

	Apart from loading state dict for `model` and `optimizer` it also updates
	`best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
	and `best_valid_loss` in `params`.

	Args:
	  params:
		The return value of :func:`get_params`.
	  model:
		The training model.
	Returns:
	  Return a dict containing previously saved training info.
	"""
	if params.start_epoch > 1:
		filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
	else:
		return None

	assert filename.is_file(), f"{filename} does not exist!"

	saved_params = load_checkpoint(filename, model=model)

	keys = [
		"best_train_epoch",
		"best_valid_epoch",
		"batch_idx_train",
		"best_train_loss",
		"best_valid_loss",
	]
	for k in keys:
		params[k] = saved_params[k]

	return saved_params


def prepare_input(batch: dict, tokenizer: Tokenizer, device: torch.device, params):
	"""Parse batch data"""
	mel_mean = params.data_args.data_statistics.mel_mean
	mel_std_inv = 1 / params.data_args.data_statistics.mel_std
	for i in range(batch["features"].shape[0]):
		n = batch["features_lens"][i]
		batch["features"][i : i + 1, :n, :] = (
			batch["features"][i : i + 1, :n, :] - mel_mean
		) * mel_std_inv
		batch["features"][i : i + 1, n:, :] = 0

	audio = batch["audio"].to(device)
	features = batch["features"].to(device)
	audio_lens = batch["audio_lens"].to(device)
	features_lens = batch["features_lens"].to(device)
	tokens = batch["tokens"]

	tokens = tokenizer.texts_to_token_ids(tokens, intersperse_blank=True)
	tokens = k2.RaggedTensor(tokens)
	row_splits = tokens.shape.row_splits(1)
	tokens_lens = row_splits[1:] - row_splits[:-1]
	tokens = tokens.to(device)
	tokens_lens = tokens_lens.to(device)
	# a tensor of shape (B, T)
	tokens = tokens.pad(mode="constant", padding_value=tokenizer.pad_id)

	max_feature_length = fix_len_compatibility(features.shape[1])
	if max_feature_length > features.shape[1]:
		pad = max_feature_length - features.shape[1]
		features = torch.nn.functional.pad(features, (0, 0, 0, pad))

		#  features_lens[features_lens.argmax()] += pad

	return audio, audio_lens, features, features_lens.long(), tokens, tokens_lens.long()


def compute_validation_loss(
	params: AttributeDict,
	model: Union[nn.Module, DDP],
	tokenizer: Tokenizer,
	valid_dl: torch.utils.data.DataLoader,
	world_size: int = 1,
	rank: int = 0,
) -> typing.Tuple[MetricsTracker, torch.Tensor]:
	"""Run the validation process with mel output."""
	model.eval()
	device = model.device if isinstance(model, DDP) else next(model.parameters()).device
	get_losses_with_output = model.module.get_losses_with_output if isinstance(model, DDP) else model.get_losses_with_output
	get_losses = model.module.get_losses if isinstance(model, DDP) else model.get_losses

	# used to summary the stats over iterations
	tot_loss = MetricsTracker()
	ret_mel = None

	with torch.no_grad():
		for batch_idx, batch in enumerate(valid_dl):
			(
				audio,
				audio_lens,
				features,
				features_lens,
				tokens,
				tokens_lens,
			) = prepare_input(batch, tokenizer, device, params)

			inputs = {
				"x": tokens,
				"x_lengths": tokens_lens,
				"y": features.permute(0, 2, 1),
				"y_lengths": features_lens,
				"spks": None,  # should change it for multi-speakers
				"durations": None,
			}
			if batch_idx == 0:
				losses:typing.Dict[str, typing.Any] = get_losses_with_output(inputs)
				ret_mel = losses['mel']
				losses.pop('mel')
			else:
				losses = get_losses(inputs)

			batch_size = len(batch["tokens"])

			loss_info = MetricsTracker()
			loss_info["samples"] = batch_size

			s = 0

			for key, value in losses.items():
				v = value.detach().item()
				loss_info[key] = v * batch_size
				s += v * batch_size

			loss_info["tot_loss"] = s

			# summary stats
			tot_loss = tot_loss + loss_info

	if world_size > 1:
		tot_loss.reduce(device)

	loss_value = tot_loss["tot_loss"] / tot_loss["samples"]
	if loss_value < params.best_valid_loss:
		params.best_valid_epoch = params.cur_epoch
		params.best_valid_loss = loss_value

	params.last_loss_value = loss_value
	params.last_loss_value_in_epoch = params.cur_epoch
	return tot_loss, ret_mel

@torch.inference_mode()
def infer_one_batch(
	writer: SummaryWriter,
	params: AttributeDict,
	valid_text: typing.List[dict],
	mel: torch.Tensor, # [2, 80, 860]
	vocoder: Union[MelDecoder, None]
):
	logging.info("Start infer one batch...")
	used_mel = mel[:len(valid_text)]
	for idx, mel in enumerate(used_mel):
		waveform = vocoder(mel).detach().cpu()
		writer.add_audio(f"valid/audio{idx}", waveform, params.batch_idx_train, sample_rate=params.data_args.sampling_rate)

class KeepBestNCheckpoints:
	def __init__(self, name: str, save_dir: str, n: int = 5):
		self.name = name
		self.n = n
		self.save_dir = save_dir
		self.checkpoints: typing.List[typing.Tuple[float, str]] = []
		
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

	def Run(self, epoch: int, metric_value: float, model_filename: str):
		"""
		must run in rank0
		"""
		checkpoint_name = f"epoch-{epoch}-{metric_value:.4f}.pt"
		checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

		heapq.heappush(self.checkpoints, (-metric_value, checkpoint_path))
		copyfile(src=model_filename, dst=checkpoint_path)
		logging.info(f"Saving {self.name} checkpoint to {checkpoint_path}")
		
		if len(self.checkpoints) > self.n:
			_, worst_checkpoint = heapq.heappop(self.checkpoints)
			os.remove(worst_checkpoint)  # Remove the worst checkpoint

class KeepLastNCheckpoints:
	def __init__(self, save_dir: str, n: int = 5):
		self.save_dir = save_dir
		self.n = n
		self.checkpoints: typing.List[str] = []
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

	def Run(self, params, model, scaler, optimizer = None, scheduler = None):
		"""
		must run in rank0
		"""
		checkpoint_filename = f"epoch-{params.cur_epoch}.pt"
		checkpoint_path = os.path.join(self.save_dir, checkpoint_filename)

		save_checkpoint(
			filename=checkpoint_path,
			params=params,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			scaler=scaler,
			rank=0
		)
		self.checkpoints.append(checkpoint_path)

		if len(self.checkpoints) > self.n:
			worst_checkpoint = self.checkpoints.pop(0)
			os.remove(worst_checkpoint)   # Remove the worst checkpoint

		return checkpoint_path

def train_one_epoch(
	params: AttributeDict,
	model: Union[nn.Module, DDP],
	tokenizer: Tokenizer,
	optimizer: Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	train_dl: torch.utils.data.DataLoader,
	valid_dl: torch.utils.data.DataLoader,
	scaler: GradScaler,
	tb_writer: Optional[SummaryWriter] = None,
	world_size: int = 1,
	rank: int = 0,
	vocoder: Union[MelDecoder, None] = None,
	valid_text: Union[typing.List[dict], None] = None,
) -> None:
	"""Train the model for one epoch.

	The training loss from the mean of all frames is saved in
	`params.train_loss`. It runs the validation process every
	`params.valid_interval` batches.

	Args:
	  params:
		It is returned by :func:`get_params`.
	  model:
		The model for training.
	  optimizer:
		The optimizer.
	  train_dl:
		Dataloader for the training dataset.
	  valid_dl:
		Dataloader for the validation dataset.
	  scaler:
		The scaler used for mix precision training.
	  tb_writer:
		Writer to write log messages to tensorboard.
	"""
	model.train()
	device = model.device if isinstance(model, DDP) else next(model.parameters()).device
	get_losses = model.module.get_losses if isinstance(model, DDP) else model.get_losses

	# used to track the stats over iterations in one epoch
	tot_loss = MetricsTracker()

	saved_bad_model = False

	def save_bad_model(suffix: str = ""):
		save_checkpoint(
			filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
			model=model,
			params=params,
			optimizer=optimizer,
			scheduler=scheduler,
			scaler=scaler,
			rank=0,
		)

	for batch_idx, batch in enumerate(train_dl):
		params.batch_idx_train += 1
		# audio: (N, T), float32
		# features: (N, T, C), float32
		# audio_lens, (N,), int32
		# features_lens, (N,), int32
		# tokens: List[List[str]], len(tokens) == N

		batch_size = len(batch["tokens"])

		(
			audio,
			audio_lens,
			features,
			features_lens,
			tokens,
			tokens_lens,
		) = prepare_input(batch, tokenizer, device, params)
		try:
			with autocast(enabled=params.use_fp16):
				losses = get_losses(
					{
						"x": tokens,
						"x_lengths": tokens_lens,
						"y": features.permute(0, 2, 1),
						"y_lengths": features_lens,
						"spks": None,  # should change it for multi-speakers
						"durations": None,
					}
				)

				loss = sum(losses.values())

				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
				scheduler.step()
				if tb_writer is not None:
					tb_writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], params.batch_idx_train)

				loss_info = MetricsTracker()
				loss_info["samples"] = batch_size

				s = 0

				for key, value in losses.items():
					v = value.detach().item()
					loss_info[key] = v * batch_size
					s += v * batch_size

				loss_info["tot_loss"] = s

				tot_loss = tot_loss + loss_info
		except:  # noqa
			save_bad_model()
			raise

		if params.batch_idx_train % 100 == 0 and params.use_fp16:
			# If the grad scale was less than 1, try increasing it.
			# The _growth_interval of the grad scaler is configurable,
			# but we can't configure it to have different
			# behavior depending on the current grad scale.
			cur_grad_scale = scaler._scale.item()

			if cur_grad_scale < 8.0 or (
				cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0
			):
				scaler.update(cur_grad_scale * 2.0)
			if cur_grad_scale < 0.01:
				if not saved_bad_model:
					save_bad_model(suffix="-first-warning")
					saved_bad_model = True
				logging.warning(f"Grad scale is small: {cur_grad_scale}")
			if cur_grad_scale < 1.0e-05:
				save_bad_model()
				raise RuntimeError(
					f"grad_scale is too small, exiting: {cur_grad_scale}"
				)

		if params.batch_idx_train % params.log_interval == 0:
			cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

			logging.info(
				f"Epoch {params.cur_epoch}, batch {batch_idx}, "
				f"global_batch_idx: {params.batch_idx_train}, "
				f"batch size: {batch_size}, "
				f"loss[{loss_info}], tot_loss[{tot_loss}], "
				+ (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
			)

			if tb_writer is not None:
				loss_info.write_summary(
					tb_writer, "train/current_", params.batch_idx_train
				)
				tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
				if params.use_fp16:
					tb_writer.add_scalar(
						"train/grad_scale", cur_grad_scale, params.batch_idx_train
					)

		if params.batch_idx_train % params.valid_interval == 1:
			logging.info("Computing validation loss")
			valid_info, mel = compute_validation_loss( # mel = (tensor([[[2, 80, 860]]]), )
				params=params,
				model=model,
				tokenizer=tokenizer,
				valid_dl=valid_dl,
				world_size=world_size,
				rank=rank,
			)
			logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
			logging.info(
				"Maximum memory allocated so far is "
				f"{torch.cuda.max_memory_allocated()//1000000}MB"
			)
			if tb_writer is not None:
				valid_info.write_summary(
					tb_writer, "train/valid_", params.batch_idx_train
				)

				if vocoder is not None:
					model_wrapper = ModelWrapper(tokenizer, model, None)
					model_wrapper.device = device
					infer_one_batch(
						writer=tb_writer,
						params=params,
						valid_text=valid_text,
						mel=mel[0],
						vocoder=vocoder
					)
					del model_wrapper
			model.train()

	loss_value = tot_loss["tot_loss"] / tot_loss["samples"]
	params.train_loss = loss_value
	if params.train_loss < params.best_train_loss:
		params.best_train_epoch = params.cur_epoch
		params.best_train_loss = params.train_loss


def run(rank, world_size, args):
	params = get_params()
	params.update(vars(args))

	fix_random_seed(params.seed)
	if world_size > 1:
		setup_dist(rank, world_size, params.master_port)

	setup_logger(f"{params.exp_dir}/log/log-train")
	logging.info("Training started")

	if args.tensorboard and rank == 0:
		tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
	else:
		tb_writer = None

	device = torch.device("cpu")
	if not IsWindows and torch.cuda.is_available(): # my windows cannot install cuda k2 library
		device = torch.device("cuda", rank)
	logging.info(f"Device: {device}")

	if rank == 0:
		saving_best_train = KeepBestNCheckpoints("best-train", str(params.exp_dir / "best-train"), args.keep_nbest_train)
		saving_best_valid = KeepBestNCheckpoints("best_valid", str(params.exp_dir / "best-valid"), args.keep_nbest_valid)
		saving_last_epoch = KeepLastNCheckpoints(str(params.exp_dir / "last-epoch"), args.keep_nepochs)
		logging.info(f"Best train checkpoints will save to {saving_best_train.save_dir}")
		logging.info(f"Best validation checkpoints will save to {saving_best_valid.save_dir}")
		logging.info(f"Last epoch checkpoints will save to {saving_last_epoch.save_dir}")

	tokenizer = Tokenizer(params.tokens)
	params.pad_id = tokenizer.pad_id
	params.vocab_size = tokenizer.vocab_size
	params.model_args.n_vocab = params.vocab_size

	with open(params.cmvn) as f:
		stats = json.load(f)
		params.data_args.data_statistics.mel_mean = stats["fbank_mean"]
		params.data_args.data_statistics.mel_std = stats["fbank_std"]

		params.model_args.data_statistics.mel_mean = stats["fbank_mean"]
		params.model_args.data_statistics.mel_std = stats["fbank_std"]

		params.data_args.sampling_rate = stats["sampling_rate"]
	params.model_args.optimizer.lr = args.learning_rate
	params.model_args.optimizer.weight_decay = args.learning_weight_decay

	logging.info(params)
	print(params)

	logging.info("About to create model")
	model = get_model(params)

	num_param = sum([p.numel() for p in model.parameters()])
	logging.info(f"Number of parameters: {num_param}")

	assert params.start_epoch > 0, params.start_epoch

	# Ref: https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
	if IsWindows:
		temp = pathlib.PosixPath
		pathlib.PosixPath = pathlib.WindowsPath

	if params.start_epoch == 1 and args.pretrained_checkpoint is not None:
		logging.info(f"Loading pretrained checkpoint {args.pretrained_checkpoint}")
		checkpoints = load_checkpoint(args.pretrained_checkpoint, model=model)
	else:
		checkpoints = load_checkpoint_if_available(params=params, model=model)
	
	if IsWindows:
		pathlib.PosixPath = temp

	model.to(device)

	melDecoder: Union[MelDecoder, None] = None
	if args.vocoder_checkpoint is not None:
		melDecoder = MelDecoder(args.vocoder_checkpoint, device)

	if world_size > 1:
		logging.info("Using DDP")
		model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	optimizer_type: str = args.learning_type.lower()
	if optimizer_type == "adamw":
		logging.info(f"Using AdamW optimizer")
		optimizer = torch.optim.AdamW(model.parameters(), **params.model_args.optimizer)
	else:
		logging.info(f"Using Adam optimizer")
		optimizer = torch.optim.Adam(model.parameters(), **params.model_args.optimizer)
	if args.scheduler_auto_peak is not None:
		total_steps = args.num_epochs * 20
		mult = 1
		last_acc_multi = args.scheduler_cos_mult
		for _ in range(args.scheduler_auto_peak - 1):
			mult = mult + last_acc_multi
			last_acc_multi = last_acc_multi * last_acc_multi
		scheduler_t0 = int(total_steps / mult)
		logging.info(f"Auto peak scheduler param: t0={scheduler_t0}, t_mult={args.scheduler_cos_mult}")
	else:
		scheduler_t0 = args.scheduler_cos_T0
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer,
		T_0=scheduler_t0,
		T_mult=args.scheduler_cos_mult,
		eta_min=args.scheduler_eta_min
	)
	# TODO: load scheduler from checkpoint

	logging.info("About to create datamodule")

	baker_zh = BakerZhTtsDataModule(args)

	train_cuts = baker_zh.train_cuts()
	train_dl = baker_zh.train_dataloaders(train_cuts)

	valid_cuts = baker_zh.valid_cuts()
	valid_dl = baker_zh.valid_dataloaders(valid_cuts)

	valid_text_batch = [ x["text"] for x in itertools.islice(iter(valid_dl), args.log_n_audio) ]
	valid_text: typing.List[dict] = []
	model_wrapper = ModelWrapper(tokenizer, model, None)
	model_wrapper.device = device
	for i in valid_text_batch:
		for j in i:
			# valid_text.append(model_wrapper.Encode(j))
			valid_text.append("PLACEHOLDER") # should remove this variable
			if len(valid_text) >= args.log_n_audio:
				break
	# if tb_writer is not None and rank == 0:
	# 	for i, item in enumerate(valid_text):
	# 		tb_writer.add_text(f"valid/text{i}", item["text"])
	del model_wrapper

	scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
	if checkpoints and "grad_scaler" in checkpoints:
		logging.info("Loading grad scaler state dict")
		scaler.load_state_dict(checkpoints["grad_scaler"])

	if world_size > 1:
		torch.distributed.barrier()
	for epoch in range(params.start_epoch, params.num_epochs + 1):
		logging.info(f"Start epoch {epoch}")
		fix_random_seed(params.seed + epoch - 1)
		if getattr(train_dl, "sampler") is not None:
			train_dl.sampler.set_epoch(epoch - 1)

		params.cur_epoch = epoch

		if tb_writer is not None:
			tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

		train_one_epoch(
			params=params,
			model=model,
			tokenizer=tokenizer,
			optimizer=optimizer,
			scheduler=scheduler,
			train_dl=train_dl,
			valid_dl=valid_dl,
			scaler=scaler,
			tb_writer=tb_writer,
			world_size=world_size,
			rank=rank,
			vocoder=melDecoder,
			valid_text=valid_text,
		)

		if rank == 0 and (epoch % params.save_every_n == 0 or epoch == params.num_epochs):
			filename = saving_last_epoch.Run(params, model, scaler, optimizer=optimizer, scheduler=scheduler)

			if hasattr(params, "last_loss_value_in_epoch") and params.last_loss_value_in_epoch == params.cur_epoch:
				saving_best_valid.Run(params.cur_epoch, params.last_loss_value, filename)

			if params.best_train_epoch == params.cur_epoch:
				saving_best_train.Run(params.cur_epoch, params.best_train_loss, filename)

	logging.info("Done!")

	if world_size > 1:
		torch.distributed.barrier()
		cleanup_dist()


def main():
	parser = get_parser()
	BakerZhTtsDataModule.add_arguments(parser)
	args = parser.parse_args()

	world_size = args.world_size
	assert world_size >= 1
	if world_size > 1:
		mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
	else:
		run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)
	main()
