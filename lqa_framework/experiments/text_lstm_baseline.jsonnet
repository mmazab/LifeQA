local config = import 'text_baseline.libsonnet';

config + {
  model+: {
    text_encoder:: {
      type: 'lstm',
      bidirectional: true,
      input_size: $.embedding_size,
      hidden_size: 100,
      num_layers: 1,
      dropout: 0.2,
    }
  }
}
