# -*- coding: utf-8 -*-
# Code from icefall

import re
from typing import List

import jieba, torch
from pypinyin import Style, lazy_pinyin, pinyin_dict, phrases_dict, pinyin_dict
from icefall.tokenizer import Tokenizer

whiter_space_re = re.compile(r"\s+")

punctuations_re = [
	(re.compile(x[0], re.IGNORECASE), x[1])
	for x in [
		("，", ","),
		("。", "."),
		("！", "!"),
		("？", "?"),
		("“", '"'),
		("”", '"'),
		("‘", "'"),
		("’", "'"),
		("：", ":"),
		("、", ","),
		("Ｂ", "逼"),
		("Ｐ", "批"),
	]
]

def normalize_white_spaces(text):
	return whiter_space_re.sub(" ", text)


def normalize_punctuations(text):
	for regex, replacement in punctuations_re:
		text = re.sub(regex, replacement, text)
	return text


def split_text(text: str) -> List[str]:
	"""
	Example input:  '你好呀，You are 一个好人。   去银行存钱？How about    you?'
	Example output: ['你好', '呀', ',', 'you are', '一个', '好人', '.', '去', '银行', '存钱', '?', 'how about you', '?']
	"""
	text = text.lower()
	text = normalize_white_spaces(text)
	text = normalize_punctuations(text)
	ans = []

	for seg in jieba.cut(text):
		if seg in ",.!?:\"'":
			ans.append(seg)
		elif seg == " " and len(ans) > 0:
			if ord("a") <= ord(ans[-1][-1]) <= ord("z"):
				ans[-1] += seg
		elif ord("a") <= ord(seg[0]) <= ord("z"):
			if len(ans) == 0:
				ans.append(seg)
				continue

			if ans[-1][-1] == " ":
				ans[-1] += seg
				continue

			ans.append(seg)
		else:
			ans.append(seg)

	ans = [s.strip() for s in ans]
	return ans


def generate_token_list() -> List[str]:
	token_set = set()

	word_dict = pinyin_dict.pinyin_dict
	i = 0
	for key in word_dict:
		if not (0x4E00 <= key <= 0x9FFF):
			continue

		w = chr(key)
		t = lazy_pinyin(w, style=Style.TONE3, tone_sandhi=True)[0]
		token_set.add(t)

	no_digit = set()
	for t in token_set:
		if t[-1] not in "1234":
			no_digit.add(t)
		else:
			no_digit.add(t[:-1])

	no_digit.add("dei")
	no_digit.add("tou")
	no_digit.add("dia")

	for t in no_digit:
		token_set.add(t)
		for i in range(1, 5):
			token_set.add(f"{t}{i}")

	ans = list(token_set)
	ans.sort()

	punctuations = list(",.!?:\"'")
	ans = punctuations + ans

	# use ID 0 for blank
	# Use ID 1 of _ for padding
	ans.insert(0, " ")
	ans.insert(1, "_")  #

	return ans

def convert_text_to_token(text:str):
	text_list = split_text(text)
	tokens = lazy_pinyin(text_list, style=Style.TONE3, tone_sandhi=True)
	return tokens

def write_lexicon(output_lexicon_filename: str):
	word_dict = pinyin_dict.pinyin_dict
	phrases = phrases_dict.phrases_dict

	i = 0
	with open(output_lexicon_filename, "w", encoding="utf-8") as f:
		for key in word_dict:
			if not (0x4E00 <= key <= 0x9FFF):
				continue

			w = chr(key)
			tokens = lazy_pinyin(w, style=Style.TONE3, tone_sandhi=True)[0]

			f.write(f"{w} {tokens}\n")

		for key in phrases:
			tokens = lazy_pinyin(key, style=Style.TONE3, tone_sandhi=True)
			tokens = " ".join(tokens)

			f.write(f"{key} {tokens}\n")

def process_text(text: str, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    text = split_text(text)
    tokens = lazy_pinyin(text, style=Style.TONE3, tone_sandhi=True)

    x = tokenizer.texts_to_token_ids([tokens])
    x = torch.tensor(x, dtype=torch.long, device=device)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {"x_orig": text, "x": x, "x_lengths": x_lengths}