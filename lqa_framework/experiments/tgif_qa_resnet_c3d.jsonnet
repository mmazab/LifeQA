local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'c3d-fc6']
  },
  iterator+: {
    batch_size: 32
  }
}
