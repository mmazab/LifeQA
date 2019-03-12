{
  dataset_reader: {
    type: 'lqa',
  },
  train_data_path: 'data/lqa_train.json',
  validation_data_path: 'data/lqa_dev.json',
  test_data_path: 'data/lqa_test.json',
  model: {
    type: 'longest_answer_baseline',
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['captions', 'num_fields'], ['question', 'num_tokens']],
    batch_size: 64,
  },
  trainer: {
    optimizer: {
        type: 'adagrad',
      },
  }
}
