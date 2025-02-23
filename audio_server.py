from flask import Flask, request, jsonify, send_file
import argparse
from infer_onnx import read_tokens, read_lexicon, normalize_text, convert_word_to_tokens, OnnxHifiGANModel, OnnxModel
from icefall.utils import intersperse
import jieba
import torch
import soundfile as sf

app = Flask(__name__)

def parse_arguments(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--acoustic-model",
		type=str,
		required=True,
		help="Path to the acoustic model",
	)

	parser.add_argument(
		"--tokens",
		type=str,
		required=True,
		help="Path to the tokens.txt",
	)

	parser.add_argument(
		"--lexicon",
		type=str,
		required=True,
		help="Path to the lexicon.txt",
	)

	parser.add_argument(
		"--vocoder",
		type=str,
		required=True,
		help="Path to the vocoder",
	)

	parser.add_argument(
		"--server",
		type=str,
		default="127.0.0.1",
	)

	parser.add_argument(
		"--port",
		type=int,
		default=41233
	)

	return parser.parse_args(args=args)

class Model:
	def __init__(self, model_filename: str, hifigan_filename: str, token_filename: str, lexicon_filename: str):
		self.token2id = read_tokens(token_filename)
		self.word2tokens = read_lexicon(lexicon_filename)
		self.model = OnnxModel(model_filename)
		self.vocoder = OnnxHifiGANModel(hifigan_filename)

	def text2tokens(self, input_text:str):
		text = normalize_text(input_text)
		seg = jieba.cut(text)
		tokens = []
		for s in seg:
			if s in self.token2id:
				tokens.append(s)
				continue

			t = convert_word_to_tokens(self.word2tokens, s)
			if t:
				tokens.extend(t)
    
		return tokens
	
	def tokens2tensor(self, tokens: list[str]):
		x = []
		for t in tokens:
			if t in self.token2id:
				x.append(self.token2id[t])

		x = intersperse(x, item=self.token2id["_"])

		x = torch.tensor(x, dtype=torch.int64).unsqueeze(0)
		return x

	def GetSampleRate(self):
		return self.model.sample_rate

	def ForwardWithTokens(self, tokens: list[str]):
		input = self.tokens2tensor(tokens)
		mel = self.model(input)
		audio = self.vocoder(mel)
		audio = audio.squeeze()
		return audio

	def __call__(self, text: str):
		tokens = self.text2tokens(text)
		return self.ForwardWithTokens(tokens)

model: Model = None

@app.route('/v1/audio/speech', methods=['POST'])
def text_to_speech():
	data = request.json
	text = data.get('input')
	voice = data.get('voice', 'zh-CN-XiaoxiaoNeural')
	
	print(f"Got text: {text}")

	audio = model(text)
	sf.write(
		"tmp.wav",
		audio,
		model.GetSampleRate(),
		"PCM_16"
	)
	return send_file("tmp.wav", mimetype=f"audio/wav")

if __name__ == '__main__':
	args = parse_arguments()
	model = Model(args.acoustic_model, args.vocoder, args.tokens, args.lexicon)
	app.run(host=args.server, port=args.port)