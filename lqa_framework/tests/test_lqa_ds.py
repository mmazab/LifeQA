import sys
sys.path.append('../../')
from allennlp.common.testing import AllenNlpTestCase
from lqa_framework.dataset_readers import LqaDatasetReader


class TestLQA_DatasetReader(AllenNlpTestCase):
	def test_read_from_file(self):
		reader = LqaDatasetReader()
		train_dataset = reader.read('data/lqa_train.json')
		dev_dataset = reader.read('data/lqa_dev.json')
		test_dataset = reader.read('data/lqa_test.json')
		return train_dataset, dev_dataset, test_dataset


dataset_reader = TestLQA_DatasetReader()
train, dev, test = dataset_reader.test_read_from_file()
