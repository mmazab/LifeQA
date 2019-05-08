#!/usr/bin/env python
import json
import os
import sys


def load_data():
	with open('lqa_train.json') as f:
		lqa_train = json.load(f)
	with open('lqa_dev.json') as f:
		lqa_dev = json.load(f)
	with open('lqa_test.json') as f:
		lqa_test = json.load(f)

	return lqa_train, lqa_dev, lqa_test


def update_captions(fid, captions, lqa_data):
	lqa_data[fid]['manual_captions'] = [{'transcript': c.strip()} for c in captions if len(c.strip()) > 0]


def rename_automatic_captions_field(lqa_data):
	for key in lqa_data:
		lqa_data[key]['automatic_captions'] = lqa_data[key]['captions']
		del lqa_data[key]['captions']


def main():
	directories = ['lqa_trans/high_quality', 'lqa_trans/low_quality']

	lqa_train, lqa_dev, lqa_test = load_data()

	rename_automatic_captions_field(lqa_train)
	rename_automatic_captions_field(lqa_dev)
	rename_automatic_captions_field(lqa_test)

	for d in directories:
		files = os.listdir(d)
		for fname in files:
			fn = os.path.join( d, fname )
			with open(fn) as f:
				captions = f.readlines()
				fid = fname[:-4]
				if fid in lqa_train: update_captions (fid, captions, lqa_train)
				elif fid in lqa_dev: update_captions (fid, captions, lqa_dev)
				elif fid in lqa_test: update_captions (fid, captions, lqa_test)

	with open('lqa_train.json', 'w') as f:
		json.dump( lqa_train, f, sort_keys=True, indent=2)
	with open('lqa_dev.json', 'w') as f:
		json.dump( lqa_dev, f, sort_keys=True, indent=2)
	with open('lqa_test.json', 'w') as f:
		json.dump( lqa_test, f, sort_keys=True, indent=2)


if __name__ == '__main__':
	main()
