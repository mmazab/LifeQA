import io
import os
import time
from datetime import timedelta
import sys
import argparse
from IPython import embed
import json
"""Transcribe the given audio file."""
#from google.cloud import speech
from google.cloud import speech_v1 as speech
#from google.cloud.speech import enums
#from google.cloud.speech import types

#need to put our API credentials in the code for authentication that are stored as Environment Variables locally.
os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

#Following line is used to save all the console outputs in a text file.
start_time = time.monotonic()


def input_file(speech_file_path):
	global content
	if os.path.isfile(speech_file_path):
		with io.open(speech_file_path, 'rb') as audio_file:
			content = audio_file.read()
			return content
	else:
		print("File doesn't exist in the directory!")
		return None


def transcribe_file(fname):
	client = speech.SpeechClient() #Initialize SpeechClient function
	#audio = types.RecognitionAudio(content = content)

	encoding = speech.enums.RecognitionConfig.AudioEncoding.FLAC
	sample_rate_hertz = 48000
	language_code = 'en-US'
	fname = '001'
	uri =	'gs://lqa_videos/{0}.flac'.format(fname)

	audio = speech.types.RecognitionAudio(uri = uri)
	config = speech.types.RecognitionConfig(	encoding = encoding, sample_rate_hertz = sample_rate_hertz, language_code = 'en-US', enable_word_time_offsets=True   )
	response = client.long_running_recognize(config, audio)
	print('Waiting for operation to complete...')
	result = response.result(timeout=10000)

	# Each result is for a consecutive portion of the audio. Iterate through
	# them to get the transcripts for the entire audio file.

	closed_captions = []

	for result in result.results:
		alternative = result.alternatives[0]
		print('-' * 20)
		print(u'Transcript: {}'.format(alternative.transcript))
		print('Confidence: {}'.format(alternative.confidence))
		
		utt = { }
		utt['transcript'] = alternative.transcript 
		utt['confidence'] = alternative.confidence 
		words = []

		for word_info in alternative.words:
			word = word_info.word
			start_time = word_info.start_time
			end_time = word_info.end_time
			s = 'Word: {}, start_time: {}, end_time: {}'.format( word,	start_time.seconds + start_time.nanos * 1e-9,	end_time.seconds + end_time.nanos * 1e-9)
			words.append( {'word':  word, 'start':  start_time.seconds + start_time.nanos * 1e-9, 'end': end_time.seconds + end_time.nanos * 1e-9} )
			print (s)

		utt['words'] =  words
		closed_captions.append(utt)

	with open('../data/closed_captions/{0}.json'.format(fname), 'w') as f:
		json.dump( closed_captions, f, sort_keys=True, indent=4)


if __name__ == '__main__':
	path = '../data/videos'
	ppath ='../data/closed_captions/'
	file_names = os.listdir(path)
	for f in file_names[:1]:
		fname = os.path.join( path,f )
		#try:
		if not os.path.isfile(ppath + f[:-4] + '.json'): 
			transcribe_file( f[:-4]  )
		#except:
		#	continue


end_time = time.monotonic()
print("Execution_Time:", timedelta(seconds = end_time - start_time))
