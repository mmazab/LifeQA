local config = import 'tgif_qa.jsonnet';

config + {
  video_channel_size: 4096,
  dataset_reader+: {
    video_features_to_load: ['c3d-fc6']
  }
}
