import os, json, argparse, torch
from lhotse import Fbank, FbankConfig
from utils.tokens import generate_token_list, convert_text_to_token, write_lexicon
from utils.dataset import RawDataset, split_train_valid_dataset
from utils.fbank import compute_cmvn

def main(args: argparse.Namespace):
	dataset = RawDataset(args)

	cut_set = dataset.ReadCutSet()
	sampling_rate = dataset.GetSamplingRate()
	if args.sample_rate is not None:
		sampling_rate = args.sample_rate
	cut_set = cut_set.resample(sampling_rate=sampling_rate)
	print(f"Sample rate: {sampling_rate}")

	os.makedirs(args.output_dir, exist_ok=True)

	feature_extractor = Fbank(FbankConfig(
		sampling_rate=sampling_rate
	))

	feature_filename = os.path.join(args.output_dir, "features")
	# Add torch code to suppress the following warning
	# WARNING:root:num_jobs is > 1 and torch's number of threads is > 1 as well: For certain configs this can result in a never ending computation. If this happens, use torch.set_num_threads(1) to circumvent this.
	torch.set_num_threads(1)
	cut_set = cut_set.compute_and_store_features(
		feature_extractor,
		feature_filename,
		num_jobs=4,
	)
	cut_set.describe()
	print(f"Features saved to {feature_filename}")

	cmvn = compute_cmvn(cut_set)
	cmvn["sampling_rate"] = sampling_rate
	stats_filename = os.path.join(args.output_dir, "cmvn.json")
	with open(stats_filename, 'w') as f:
		json.dump(cmvn, f, indent = 4)
	print(f"Stats saved to {stats_filename}")

	token_list = generate_token_list()
	token_filename = os.path.join(args.output_dir, "tokens.txt")
	with open(token_filename, "w", encoding="utf-8") as f:
		for indx, token in enumerate(token_list):
			f.write(f"{token} {indx}\n")
	print(f"Tokens saved to {token_filename}")

	lexicon_filename = os.path.join(args.output_dir, "lexicon.txt")
	write_lexicon(lexicon_filename)
	print(f"Lexicon saved to {lexicon_filename}")

	# Convert text to token
	for cut in cut_set:
		tokens = convert_text_to_token(cut.supervisions[0].text)
		cut.tokens = tokens
		cut.supervisions[0].normalized_text = cut.supervisions[0].text

	if args.validset_ratio is None:
		dataset_filename = os.path.join(args.output_dir, f"{args.dataset_name}_train.jsonl.gz")
		cut_set.to_file(dataset_filename)
		print(f"Dataset saved to {dataset_filename}")
	else:
		train_cut_set, valid_cut_set = split_train_valid_dataset(cut_set, args.validset_ratio, seed=args.seed)
		print(f"Dataset has {len(train_cut_set)} train samples and {len(valid_cut_set)} test samples.")

		train_dataset_filename = os.path.join(args.output_dir, f"{args.dataset_name}_train.jsonl.gz")
		valid_dataset_filename = os.path.join(args.output_dir, f"{args.dataset_name}_valid.jsonl.gz")
		train_cut_set.to_file(train_dataset_filename)
		print(f"Train dataset saved to {train_dataset_filename}")
		valid_cut_set.to_file(valid_dataset_filename)
		print(f"Test dataset saved to {valid_dataset_filename}")

def parse_args(args = None):
	# Setup argument parsing
	parser = argparse.ArgumentParser(description="Trun a raw dataset to a usable dataset.")
	parser.add_argument("--output_dir", type=str, required=True, help="The output dir of dataset")
	parser.add_argument("--sample_rate", type=int, default=None, help="The sample rate of wav files")
	parser.add_argument("--validset_ratio", type=float, default=None, help="The ratio of validation set")
	parser.add_argument("--dataset_name", type=str, default="baker_zh_cuts", help="The name of dataset")
	parser.add_argument("--seed", type=int, help="Random seed for data splitting.")

	RawDataset.ParseArgs(parser)
	return parser.parse_args(args)

if __name__ == "__main__":
	args = parse_args()
	main(args)