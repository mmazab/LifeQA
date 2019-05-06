local params = import 'text_baseline.libsonnet';

params + {
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
