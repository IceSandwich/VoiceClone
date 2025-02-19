import argparse, torch, os
from icefall.tts_datamodule import BakerZhTtsDataModule
import utils
import utils.model
import soundfile as sf

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'
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
		help="Path to the non-raw dataset directory.",
	)
	parser.add_argument(
        "--vocoder",
        type=str,
		required=True,
        help="Path to the vocoder",
    )
	parser.add_argument(
		"--text",
		type=str,
		required=True,
		help="The text to be synthesized.",
	)
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		help="The output audio file.",
	)
	parser.add_argument(
        "--sampling_rate",
        type=int,
        default=22050,
        help="The sampling rate of the generated speech (default: 22050 for baker_zh)",
    )

	BakerZhTtsDataModule.add_arguments(parser)
	return parser.parse_args(args)

@torch.inference_mode()
def main(args):
	builder = utils.model.ModelBuilder()
	decoder = utils.model.MelDecoder(args.vocoder)

	token_filename = os.path.join(args.dataset_dir, 'tokens.txt')
	builder.LoadTokenizer(token_filename)

	cmvn_filename = os.path.join(args.dataset_dir, 'cmvn.json')
	builder.LoadCMVN(cmvn_filename)

	model = builder.BuildModel()
	model.LoadCheckpoint(args.model)
	model.UploadToDevice(device)
	decoder.UploadToDevice(device)

	output = model(args.text)
	waveform = decoder(output["mel"])

	sf.write(
		file=args.output,
		data=waveform,
		samplerate=args.sampling_rate,
		subtype="PCM_16",
	)
	print("Done.")

if __name__ == '__main__':
	parser = parse_args()
	main(parser)