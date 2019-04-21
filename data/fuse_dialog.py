import os
import sys
import json
from IPython import embed




def load_data():
	lqa_train, lqa_dev, lqa_test = None, None, None
	
	with open('lqa_train.json','r') as f:
		lqa_train = json.load(f)
	with open('lqa_dev.json','r') as f:
		lqa_dev = json.load(f)
	with open('lqa_test.json','r') as f:
		lqa_test = json.load(f)

	return lqa_train, lqa_dev, lqa_test



def update_captions(fid, captions, lqa_data):	
	video = lqa_data[fid]
	video ['automatic_captions'] = video['captions']
	video['captions'] = [ {'transcript': c.strip() } for c in  captions if len(c.strip()) > 0 ]
	lqa_data[fid] = video
		



def main():
	directories = ['lqa_trans/high_quality', 'lqa_trans/low_quality']



	lqa_train, lqa_dev, lqa_test = load_data()
	for d in directories:
		files = os.listdir(d)
		for fname in files:
			fn = os.path.join( d, fname )	
			with open(fn, 'r') as f:
				captions = f.readlines()
				fid = fname[:-4]
				if fid in lqa_train: update_captions (fid, captions, lqa_train)
				elif fid in lqa_dev: update_captions (fid, captions, lqa_dev)
				elif fid in lqa_test: update_captions (fid, captions, lqa_test)

	with open('lqa_train.json', 'w') as f:
		json.dump( lqa_train, f, sort_keys=True, indent=4)	
	with open('lqa_dev.json', 'w') as f:
		json.dump( lqa_dev, f, sort_keys=True, indent=4)	
	with open('lqa_test.json', 'w') as f:
		json.dump( lqa_test, f, sort_keys=True, indent=4)	

	


if __name__=='__main__':
	main()
