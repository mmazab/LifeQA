(import 'tgif_qa.jsonnet') + {
  video_channel_size: 2048 + 1024,
  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'i3d-avg-pool']
  }
}
