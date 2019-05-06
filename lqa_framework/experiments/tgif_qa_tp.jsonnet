local config = import 'tgif_qa.jsonnet';

config + {
  model+: {
    temporal_attention: {
      type: 'mlp',
      matrix_size: $.encoder.output_size / $.encoder.num_layers,
      vector_size: $.encoder.output_size,
    }
  }
}
