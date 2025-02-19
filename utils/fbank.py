from lhotse import CutSet
import torch, json

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