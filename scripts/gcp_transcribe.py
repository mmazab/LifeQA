"""Transcribe the given audio file."""
from datetime import timedelta
import json
import os
import time

from google.cloud import speech_v1 as speech


def transcribe_file(filename):
    client = speech.SpeechClient()

    encoding = speech.enums.RecognitionConfig.AudioEncoding.FLAC
    sample_rate_hertz = 48000
    uri = 'gs://lqa_videos/{0}.flac'.format(filename)

    audio = speech.types.RecognitionAudio(uri=uri)
    config = speech.types.RecognitionConfig(encoding=encoding, sample_rate_hertz=sample_rate_hertz,
                                            language_code='en-US', enable_word_time_offsets=True)
    response = client.long_running_recognize(config, audio)
    print('Waiting for operation to complete...')
    result = response.result(timeout=10000)

    # Each result is for a consecutive portion of the audio.
    # Iterate through them to get the transcripts for the entire audio file.

    closed_captions = []

    for result in result.results:
        alternative = result.alternatives[0]
        print('-' * 20)
        print(u'Transcript: {}'.format(alternative.transcript))
        print('Confidence: {}'.format(alternative.confidence))

        utt = {'transcript': alternative.transcript, 'confidence': alternative.confidence}
        words = []

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            s = 'Word: {}, start_time: {}, end_time: {}'.format(word, start_time.seconds + start_time.nanos * 1e-9,
                                                                end_time.seconds + end_time.nanos * 1e-9)
            words.append({'word': word, 'start': start_time.seconds + start_time.nanos * 1e-9,
                          'end': end_time.seconds + end_time.nanos * 1e-9})
            print(s)

        utt['words'] = words
        closed_captions.append(utt)

    with open(f'data/closed_captions/{filename}.json', 'w') as file:
        json.dump(closed_captions, file, sort_keys=True, indent=2)


def main():
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ

    # Following line is used to save all the console outputs in a text file.
    start_time = time.monotonic()

    path = 'data/videos'
    captions_path = 'data/closed_captions/'
    filenames = os.listdir(path)
    for filename in filenames[:1]:
        if not os.path.isfile(captions_path + filename[:-4] + '.json'):
            transcribe_file(filename[:-4])

    end_time = time.monotonic()
    print("Execution_time:", timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
