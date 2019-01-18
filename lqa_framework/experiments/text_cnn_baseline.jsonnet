local config = import 'text_baseline.libsonnet';

config + {
  model+: {
    text_encoder:: {
      type: 'cnn',
      num_filters: 100,
      ngram_filter_sizes: [2, 3],
      embedding_dim: $.embedding_size,
    }
  }
}
