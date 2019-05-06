(import 'text_only.libsonnet') + {
  model+: {
    text_encoder:: {
      type: 'lstm',
      bidirectional: true,
      input_size: $.embedding_size,
      hidden_size: 100,
      num_layers: 1,
    }
  }
}
