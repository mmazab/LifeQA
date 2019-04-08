local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    video_features_to_load: 'i3d'
  },
  model+: {
    video_encoder+: {
      input_size: 1024
    }
  }
}
