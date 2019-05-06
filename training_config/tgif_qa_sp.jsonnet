(import 'tgif_qa.jsonnet') + {
  video_channel_size: 2048 + 1024,
  dataset_reader+: {
    video_features_to_load: ['resnet-res5c', 'c3d-conv5b']
  },
  model+: {
    spatial_attention: {
      type: 'mlp',
      matrix_size: $.video_channel_size,
      vector_size: $.encoder.output_size,
    }
  }
}
