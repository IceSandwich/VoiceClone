import argparse, torch, os
import utils
import utils.model
import soundfile as sf
from tqdm import tqdm
from icefall.tts_datamodule import BakerZhTtsDataModule
from lhotse.cut import Cut

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))

def parse_args(args = None):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model",
		type=str,
		required=True,
		help="Path to the model file.",
	)
	parser.add_argument(
		"--dataset_dir",
		type=str,
		required=True,
		help="Path to the dataset directory.",
	)
	parser.add_argument(
		"--vocoder",
		type=str,
		required=True,
		help="Path to the vocoder",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
	)

	return parser.parse_args(args)

@torch.inference_mode()
def main(args):
	builder = utils.model.ModelBuilder()
	decoder = utils.model.MelDecoder(args.vocoder, device)

	token_filename = os.path.join(args.dataset_dir, 'tokens.txt')
	builder.LoadTokenizer(token_filename)

	cmvn_filename = os.path.join(args.dataset_dir, 'cmvn.json')
	builder.LoadCMVN(cmvn_filename)
	sampling_rate = builder.GetSamplingRate()
	if sampling_rate is None:
		print(f"Sampling rate assumes as 22050")
		sampling_rate = 22050
	else:
		print(f"Using sampling rate: {sampling_rate}")

	model = builder.BuildModel()
	model.LoadCheckpoint(args.model)
	model.UploadToDevice(device)
	# decoder.UploadToDevice(device)

	datamodule_parser = argparse.ArgumentParser()
	BakerZhTtsDataModule.add_arguments(datamodule_parser)
	datamodule_args = datamodule_parser.parse_args([
		"--manifest-dir", args.dataset_dir
	])
	baker_zh = BakerZhTtsDataModule(datamodule_args)
	cut_set = baker_zh.valid_cuts()

	os.makedirs(args.output_dir, exist_ok=True)
	
	for cut in tqdm(cut_set):
		cut: Cut
		text = cut.supervisions[0].text

		output = model(text)
		waveform = decoder(output["mel"])

		filename = os.path.join(args.output_dir, f"{text}.wav")
		sf.write(
			file=filename,
			data=waveform,
			samplerate=sampling_rate,
			subtype="PCM_16",
		)
	print("Done.")

if __name__ == '__main__':
	parser = parse_args()
	main(parser)