(import 'tgif_qa.jsonnet') + {
  video_channel_size: 2048 + 2048,
  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'resof']
  }
}
