#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script exports a Matcha-TTS model to ONNX.
Note that the model outputs fbank. You need to use a vocoder to convert
it to audio. See also ./export_onnx_hifigan.py

python3 ./matcha/export_onnx.py \
  --exp-dir ./matcha/exp-1 \
  --epoch 2000 \
  --tokens ./data/tokens.txt \
  --cmvn ./data/fbank/cmvn.json

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
import os
import shutil

import onnx
import torch
from icefall.tokenizer import Tokenizer
from train import get_model, get_params

from icefall.checkpoint import load_checkpoint


def get_parser():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	
	parser.add_argument(
		"--checkpoint",
		type=str,
		required=True,
	)

	parser.add_argument(
		"--dataset_dir",
		type=str,
		required=True
	)

	parser.add_argument(
		"--output_dir",
		type=str,
		required=True
	)
	
	return parser


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
	"""Add meta data to an ONNX model. It is changed in-place.

	Args:
	  filename:
		Filename of the ONNX model to be changed.
	  meta_data:
		Key-value pairs.
	"""
	model = onnx.load(filename)

	while len(model.metadata_props):
		model.metadata_props.pop()

	for key, value in meta_data.items():
		meta = model.metadata_props.add()
		meta.key = key
		meta.value = str(value)

	onnx.save(model, filename)


class ModelWrapper(torch.nn.Module):
	def __init__(self, model, num_steps: int = 5):
		super().__init__()
		self.model = model
		self.num_steps = num_steps

	def forward(
		self,
		x: torch.Tensor,
		x_lengths: torch.Tensor,
		noise_scale: torch.Tensor,
		length_scale: torch.Tensor,
	) -> torch.Tensor:
		"""
		Args: :
		  x: (batch_size, num_tokens), torch.int64
		  x_lengths: (batch_size,), torch.int64
		  noise_scale: (1,), torch.float32
		  length_scale (1,), torch.float32
		Returns:
		  audio: (batch_size, num_samples)

		"""
		mel = self.model.synthesise(
			x=x,
			x_lengths=x_lengths,
			n_timesteps=self.num_steps,
			temperature=noise_scale,
			length_scale=length_scale,
		)["mel"]
		# mel: (batch_size, feat_dim, num_frames)

		return mel


@torch.inference_mode()
def main():
	parser = get_parser()
	args = parser.parse_args()
	params = get_params()

	params.update(vars(args))

	token_filename = os.path.join(args.dataset_dir, "tokens.txt")
	tokenizer = Tokenizer(token_filename)
	params.pad_id = tokenizer.pad_id
	params.vocab_size = tokenizer.vocab_size
	params.model_args.n_vocab = params.vocab_size

	cmvn_filename = os.path.join(args.dataset_dir, "cmvn.json")
	with open(cmvn_filename) as f:
		stats = json.load(f)
		params.data_args.data_statistics.mel_mean = stats["fbank_mean"]
		params.data_args.data_statistics.mel_std = stats["fbank_std"]

		params.model_args.data_statistics.mel_mean = stats["fbank_mean"]
		params.model_args.data_statistics.mel_std = stats["fbank_std"]

		assert stats["sampling_rate"] == 22050, "only support sampling rate 22050 right now. use change_sample_rate.py to adjust your dataset."
	logging.info(params)

	logging.info("About to create model")
	model = get_model(params)
	load_checkpoint(args.checkpoint, model)

	os.makedirs(args.output_dir, exist_ok=True)

	for num_steps in [2, 3, 4, 5, 6]:
		logging.info(f"num_steps: {num_steps}")
		wrapper = ModelWrapper(model, num_steps=num_steps)
		wrapper.eval()

		# Use a large value so the rotary position embedding in the text
		# encoder has a large initial length
		x = torch.ones(1, 1000, dtype=torch.int64)
		x_lengths = torch.tensor([x.shape[1]], dtype=torch.int64)
		noise_scale = torch.tensor([1.0])
		length_scale = torch.tensor([1.0])

		opset_version = 14
		filename = os.path.join(args.output_dir, f"model-steps-{num_steps}.onnx")
		torch.onnx.export(
			wrapper,
			(x, x_lengths, noise_scale, length_scale),
			filename,
			opset_version=opset_version,
			input_names=["x", "x_length", "noise_scale", "length_scale"],
			output_names=["mel"],
			dynamic_axes={
				"x": {0: "N", 1: "L"},
				"x_length": {0: "N"},
				"mel": {0: "N", 2: "L"},
			},
		)

		meta_data = {
			"model_type": "matcha-tts",
			"language": "Chinese",
			"has_espeak": 0,
			"n_speakers": 1,
			"jieba": 1,
			"sample_rate": 22050,
			"version": 1,
			"pad_id": params.pad_id,
			"model_author": "icefall",
			"maintainer": "k2-fsa",
			"dataset": "baker-zh",
			"use_eos_bos": 0,
			"dataset_url": "https://www.data-baker.com/open_source.html",
			"dataset_comment": "The dataset is for non-commercial use only.",
			"num_ode_steps": num_steps,
		}
		add_meta_data(filename=filename, meta_data=meta_data)
		print(meta_data)

	# hifigan_onnx_filename = Path(__file__).parent / "assets" / "model" / "vocoder" / "hifigan_v2.onnx"
	# target_onnx_filename = os.path.join(args.output_dir, "hifigan.onnx")
	# shutil.copyfile(hifigan_onnx_filename, target_onnx_filename)
	# logging.info(f"Copy hifigan from {hifigan_onnx_filename} to {target_onnx_filename}")

	target_token_filename = os.path.join(args.output_dir, "tokens.txt")
	shutil.copyfile(token_filename, target_token_filename)
	logging.info(f"Copy tokens from {token_filename} to {target_token_filename}")


if __name__ == "__main__":
	formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

	logging.basicConfig(format=formatter, level=logging.INFO)
	main()
