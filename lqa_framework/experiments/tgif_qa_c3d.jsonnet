local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    video_features_to_load: 'c3d'
  },
  model+: {
    video_encoder+: {
      input_size: 4096
    }
  },
  iterator+: {
    batch_size: 32
  }
}
