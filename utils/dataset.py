import os
import argparse
import typing
import random
from lhotse import Recording, RecordingSet, SupervisionSegment, CutSet
from lhotse.cut import Cut
from lhotse.supervision import SupervisionSet
from .tokens import convert_text_to_token, normalize_punctuations
import re
import time

# 数据集下是：
# wav_folder
# - 1.wav
# - 2.wav
# - ...
# - 100.wav
# wav的名字是字幕的行号

def contains_alphanumeric(s):
    # 使用正则表达式检查字符串是否包含阿拉伯数字或英文字母
    return bool(re.search(r'[a-zA-Z0-9]', s))

class RawDataset:
	def __init__(self, args):
		self.args = args

	@classmethod
	def ParseArgs(cls, parser: argparse.ArgumentParser):
		parser.add_argument_group(
			title="TTS raw dataset"
		)
		parser.add_argument(
			"--wav_folder", 
			type=str, 
			required=True, 
			help="Path to the folder containing .wav files."
		)
		parser.add_argument(
			"--label_text", 
			type=str, 
			required=True, 
			help="The text file of wavs."
		)
		parser.add_argument(
			"--language", 
			type=str, 
			help="The language of sounds"
		)
		parser.add_argument(
			"--speaker", 
			type=str, 
			help="The speaker of sounds"
		)
		parser.add_argument(
			"--gender", 
			type=str, 
			help="The gender of speaker"
		)
		parser.add_argument(
			"--skip_alphanumeric",
			type=bool,
			default=False,
		)

	def read(self):
		audio_filename: typing.List[str] = [f for f in os.listdir(self.args.wav_folder) if f.endswith('.wav')]
		audio_filename = sorted(audio_filename, key=lambda x: int(x[:-4]))
		audio_path = [ os.path.join(self.args.wav_folder, f) for f in audio_filename ]
		ret = {
			"audio_filename": audio_filename,
			"audio_path": audio_path,
		}

		self.isv2 = 'v2' in os.path.basename(self.args.label_text)

		if self.isv2:
			print(f"The label file is v2 format. If it's not, please don't include v2 in its name.")
			def parse_v2_dataset(filename: str):
				dataset: typing.List[typing.Tuple[str, str]] = []
				with open(filename, 'r', encoding='utf-8') as f:
					lines = f.readlines()
					for i in range(0, len(lines), 2):
						text = lines[i].strip().strip('\n')
						phome = lines[i+1].strip().strip('\n')
						dataset.append((text, phome))
				return dataset
			dataset = parse_v2_dataset(self.args.label_text)
			transcription_lines = [ x[0] for x in dataset ]
			tokens = [ x[1] for x in dataset ]
			ret['transcription_lines'] = transcription_lines
			ret['tokens'] = tokens
		else:
			with open(self.args.label_text, 'r', encoding='utf-8') as f:
				transcription_lines = [ x.strip() for x in f.readlines() if x.strip() != "" ]
				if len(audio_filename) != len(transcription_lines):
					print(f"[Error] The number of audio is not same as the transcript!!!")
					print(f"[Error] The number of audio in folder({self.args.wav_folder}) is {len(audio_filename)}.")
					print(f"[Error] The non-blank lines of transcript of file({self.args.label_text}) is {len(transcription_lines)}.")
					raise Exception("the number of audio and transcript doesn't match.")
			print("Got audio files:", len(audio_filename))
			ret["transcription_lines"] = transcription_lines
		return ret
	
	def ReadCutSet(self):
		readinfo = self.read()

		recordings: typing.List[Recording] = []
		segments: typing.List[SupervisionSegment] = []
		self.sampling_rate = 100000

		for idx, (audio_path, audio_filename, transcription) in enumerate(zip(readinfo["audio_path"], readinfo["audio_filename"], readinfo["transcription_lines"])):
			if self.args.skip_alphanumeric and contains_alphanumeric(transcription):
				print(f"Skip {transcription}")
				continue

			# Create a Recording object
			recording = Recording.from_file(audio_path)
			self.sampling_rate = min(self.sampling_rate, recording.sampling_rate)	
			recordings.append(recording)

			# Create a SupervisionSegment object
			segment = SupervisionSegment(
				id=recording.id + "-seg",
				recording_id=recording.id,
				start=0.0,  # You may want to adjust the start and end times based on your use case
				duration=recording.duration,
				text=transcription,
				language=self.args.language,
				speaker=self.args.speaker,
				gender=self.args.gender,
			)
			segments.append(segment)

		recording_set = RecordingSet.from_recordings(recordings)
		supervision_set = SupervisionSet.from_segments(segments)
		cut_set = CutSet.from_manifests(
			recordings=recording_set,
			supervisions=supervision_set
		)

		if self.isv2:
			mapping = { k: v for (k, v) in zip(readinfo["transcription_lines"], readinfo["tokens"]) }
			for cut in cut_set:
				cut: Cut
				cut.tokens = normalize_punctuations(mapping[cut.supervisions[0].text]).split(' ')
				cut.supervisions[0].normalized_text = normalize_punctuations(cut.supervisions[0].text)
		else:
			for cut in cut_set:
				tokens = convert_text_to_token(cut.supervisions[0].text)
				cut.tokens = tokens
				cut.supervisions[0].normalized_text = cut.supervisions[0].text

		return cut_set
	
	def IsV2(self):
		return self.isv2
	
	def GetV2TokenFilename(self):
		return os.path.splitext(self.args.label_text)[0] + "_tokens.txt"
	
	def GetSamplingRate(self) -> float:
		if not hasattr(self,'sampling_rate'):
			self.ReadCutSet()
		return self.sampling_rate

def split_train_valid_dataset(dataset: CutSet, validation_ratio: float, seed: int = None) -> typing.Tuple[CutSet, CutSet]:
	"""Split a dataset into train and valid sets."""
	if seed is None:
		seed = int(time.time())
	random.seed(seed)
	print(f"Using seed {seed}")

	length = len(dataset)
	sample_indexs = list(range(length))
	random.shuffle(sample_indexs)
	valid_samples_ids = random.sample(sample_indexs, int(validation_ratio * length))

	valid_segment_ids = [ dataset[i].supervisions[0].id for i in valid_samples_ids ]
	train_segment_ids = [ dataset[i].supervisions[0].id for i in range(length) if i not in valid_samples_ids ]

	train_cut_set = dataset.subset(supervision_ids=train_segment_ids)
	valid_cut_set = dataset.subset(supervision_ids=valid_segment_ids)

	return train_cut_set, valid_cut_set
