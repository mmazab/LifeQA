local config = import 'tgif_qa.jsonnet';

config + {
  video_channel_size: 1024,
  dataset_reader+: {
    video_features_to_load: ['i3d-avg-pool']
  }
}
