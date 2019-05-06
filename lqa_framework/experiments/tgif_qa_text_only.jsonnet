local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    video_features_to_load: null
  },
  model+: {
    video_encoder: null,
    text_video_mode: 'text'
  },
  iterator: {
    type: 'basic',
    batch_size: 8,
  }
}
