import os
import argparse
import typing
import random
from lhotse import Recording, RecordingSet, SupervisionSegment, CutSet
from lhotse.supervision import SupervisionSet
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

	def read(self):
		audio_filename: typing.List[str] = [f for f in os.listdir(self.args.wav_folder) if f.endswith('.wav')]
		audio_path = [ os.path.join(self.args.wav_folder, f) for f in audio_filename ]
		with open(self.args.label_text, 'r', encoding='utf-8') as f:
			transcription_lines = [ x.strip() for x in f.readlines() if x.strip() != "" ]
			if len(audio_filename) != len(transcription_lines):
				print(f"[Error] The number of audio is not same as the transcript!!!")
				print(f"[Error] The number of audio in folder({self.args.wav_folder}) is {len(audio_filename)}.")
				print(f"[Error] The non-blank lines of transcript of file({self.args.label_text}) is {len(transcription_lines)}.")
				raise Exception("the number of audio and transcript doesn't match.")
		print("Got audio files:", len(audio_filename))
		return audio_path, audio_filename, transcription_lines
	
	def ReadCutSet(self):
		audio_paths, audio_filenames, transcription_lines = self.read()

		recordings: typing.List[Recording] = []
		segments: typing.List[SupervisionSegment] = []
		self.sampling_rate = 100000

		for idx, (audio_path, audio_filename, transcription) in enumerate(zip(audio_paths, audio_filenames, transcription_lines)):
			if contains_alphanumeric(transcription):
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

		return cut_set
	
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
	range_index = random.shuffle(list(range(length)))
	valid_samples_ids = random.sample(range_index, int(validation_ratio * length))

	valid_segment_ids = [ dataset[i].supervisions[0].id for i in valid_samples_ids ]
	train_segment_ids = [ dataset[i].supervisions[0].id for i in range(length) if i not in valid_samples_ids ]

	train_cut_set = dataset.subset(supervision_ids=train_segment_ids)
	valid_cut_set = dataset.subset(supervision_ids=valid_segment_ids)

	return train_cut_set, valid_cut_set
