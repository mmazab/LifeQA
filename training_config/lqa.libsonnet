{
  dataset_reader: {
    type: 'lqa'
  },
  train_data_path: 'data/lqa_train.json',
  validation_data_path: 'data/lqa_dev.json',
  test_data_path: 'data/lqa_test.json', // Comment this line if not used, so experiments run faster.
  evaluate_on_test: true,
  trainer: {
    type: 'cross_validation',
    splitter: {
      type: 'group_k_fold',
      n_splits: 5,
      generate_validation_sets: true,
    },
    group_key: 'parent_video_id',
  }
}
