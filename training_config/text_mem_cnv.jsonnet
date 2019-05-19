(import 'text_only_mem.libsonnet') + {
  dataset_reader+: {
    token_indexers+: {
      tokens+: {
        token_min_padding_length: 3
      }
    }
  },
  model+: {
    text_encoder:: {
      type: 'cnn',
      num_filters: 100,
      ngram_filter_sizes: [1, 2, 3],
      embedding_dim: $.embedding_size,
    }
  }
}
