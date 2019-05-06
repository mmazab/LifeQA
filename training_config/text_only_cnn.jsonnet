(import 'text_only.libsonnet') + {
  model+: {
    text_encoder:: {
      type: 'cnn',
      num_filters: 100,
      ngram_filter_sizes: [2, 3],
      embedding_dim: $.embedding_size,
    }
  }
}
