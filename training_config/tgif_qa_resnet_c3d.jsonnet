(import 'tgif_qa.jsonnet') + {
  video_channel_size: 2048 + 4096,
  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'c3d-fc6']
  }
}
