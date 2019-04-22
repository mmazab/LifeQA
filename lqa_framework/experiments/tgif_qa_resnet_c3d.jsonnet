local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'c3d-fc6']
  },
  model+: {
    video_encoder+: {
      input_size: 2048 + 4096
    }
  }
}
