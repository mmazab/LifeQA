{
  dataset_reader: {
    type: 'lqa'
  },
  train_data_path: 'data/folds/fold0_train.json',
  validation_data_path: 'data/folds/fold0_test.json',
  //train_data_path: 'data/lqa_train.json',
  //validation_data_path: 'data/lqa_dev.json',
  //test_data_path: 'data/lqa_test.json', // Don't load test data for now so experiments are faster (it's not used).
}
