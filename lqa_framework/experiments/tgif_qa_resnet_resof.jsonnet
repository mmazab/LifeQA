local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'resof']
  },
  model+: {
    video_encoder+: {
      input_size: 2048 + 2048
    }
  },
  iterator+: {
    batch_size: 32
  }
}
