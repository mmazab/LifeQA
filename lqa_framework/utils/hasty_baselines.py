import os
import csv
import numpy as np
from IPython import embed
import random
import json


def run():
	fname = '../../data/questions/combined_question.csv'
	path  = '../../data/closed_captions/'
	train, test, dev = [], [], []
	correct, correct_index = [], []

	questions = {}
	qid = 0
	
	with open(fname, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',')
		next(csv_reader, None)	
		for row in csv_reader:
			correct = row[2] 
			answers = [	row[2], row[3], row[4], row[5] ]
			random.shuffle( answers )
			correct_index.append( answers.index(correct ) )
			train.append( [ len(answers[0]), len(answers[1]), len(answers[2]), len(answers[3]) ] )	
			qid+=1
			questions[ str(row[0]).zfill(3)] = questions.get(str(row[0]).zfill(3), {'captions': None, 'questions':[]})
			questions[ str(row[0]).zfill(3)]['questions'].append({'q_id': qid, 'question': row[1] , 'answers': [ answers[0], answers[1], answers[2], answers[3] ], 'correct_index': correct_index[-1] })
			

	data = np.asarray(train)
	longest = np.zeros(len(train))
	shortest = np.zeros(len(train))
	longest, shortest = np.argmax(data, axis=1), np.argmin(data,axis=1)
	
	l, s = float(np.sum(longest == correct_index))/len(train), float (np.sum(shortest == correct_index)) /len(train)
	print (l, s)

	for cf in questions:
		fname = str(cf).zfill(3)
		if os.path.isfile(path + fname + '.json'):
			with open(path + fname  + '.json','r') as f:
				captions = json.load(f)
				questions[cf]['captions'] = captions
	
	with open('../../data/lqa_data.json', 'w') as f:
		json.dump(questions, f, sort_keys=True, indent=4)
		print ('Data written to file...')


def main():
	run()



if __name__=='__main__':
	main()
