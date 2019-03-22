local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    load_video_features: false
  },
  model+: {
    video_encoder: null,
    text_video_mode: 'text'
  }
}
