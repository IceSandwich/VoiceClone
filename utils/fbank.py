from lhotse import CutSet
import torch, json

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from icefall.fbank import MatchaFbank, MatchaFbankConfig

def get_feature_extractor():
	config = MatchaFbankConfig(
        n_fft=1024,
        n_mels=80,
        sampling_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
    )
	extractor = MatchaFbank(config)
	return extractor

# Calcute fbank mean and std.
def compute_cmvn(cut_set: CutSet):
	feat_dim = cut_set[0].features.num_features
	num_frames = 0
	s = 0
	sq = 0
	for c in cut_set:
		f = torch.from_numpy(c.load_features())
		num_frames += f.shape[0]
		s += f.sum()
		sq += f.square().sum()

	fbank_mean = s / (num_frames * feat_dim)
	fbank_var = sq / (num_frames * feat_dim) - fbank_mean * fbank_mean
	fbank_std = fbank_var.sqrt()
	return {"fbank_mean": fbank_mean.item(), "fbank_std": fbank_std.item()}

def ReadCMVN(cmvn_filename: str):
	with open(cmvn_filename, 'r') as f:
		stats = json.load(f)
		return stats