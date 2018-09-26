import sys
sys.path.append('../../')
from allennlp.common.testing import AllenNlpTestCase
from lqa_framework.dataset_readers import LqaTextDatasetReader
from IPython import embed



class TestLQA_DatasetReader(AllenNlpTestCase):
	def test_read_from_file(self):
		reader = LqaTextDatasetReader()
		train_dataset, dev_dataset, test_dataset = None, None, None
		train_dataset = reader.read('/scratch/mihalcea_fluxg/mazab/lifeqa/data/lqa_train.json')
		dev_dataset = reader.read('/scratch/mihalcea_fluxg/mazab/lifeqa/data/lqa_dev.json')
		test_dataset = reader.read('/scratch/mihalcea_fluxg/mazab/lifeqa/data/lqa_test.json')
		return train_dataset, dev_dataset, test_dataset


dataset_reader = TestLQA_DatasetReader()
train, dev, test = dataset_reader.test_read_from_file()

