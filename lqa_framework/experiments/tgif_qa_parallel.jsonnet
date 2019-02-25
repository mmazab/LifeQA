local config = import 'tgif_qa.jsonnet';

config + {
  model+: {
    text_video_mode: 'parallel'
  }
}
