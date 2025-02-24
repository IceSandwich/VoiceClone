import argparse
import os
from tqdm import tqdm

def main(args):
	os.makedirs(args.output_folder, exist_ok=True)
	for x in tqdm(os.listdir(args.wav_folder)):
		srcFilename = os.path.join(args.wav_folder, x)
		dstFilename = os.path.join(args.output_folder, x)
		os.system(f'ffmpeg -i {srcFilename} -ar {args.sample_rate} -y {dstFilename}')

def parse_args(args = None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--wav_folder', type = str, required=True)
	parser.add_argument('--output_folder', type = str, required= True)
	parser.add_argument('--sample_rate', type = int, default = 22050)
	return parser.parse_args(args)

if __name__ == '__main__':
	args = parse_args()
	main(args)