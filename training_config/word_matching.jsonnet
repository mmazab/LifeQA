(import 'simple.libsonnet') + {
  dataset_reader+: {
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    }
  },
  model: 'word_matching'
}
