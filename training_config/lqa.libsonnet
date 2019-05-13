{
  dataset_reader: {
    type: 'lqa'
  },
  train_data_path: 'data/lqa_train.json',
  validation_data_path: 'data/lqa_dev.json',
  //test_data_path: 'data/lqa_test.json', // Don't load test data for now so experiments are faster (it's not used).
  trainer: {
    type: 'cross_validation',
    splitter: {
      type: 'group_k_fold',
      n_splits: 5,
    },
    group_key: 'parent_video_id',
  }
}
