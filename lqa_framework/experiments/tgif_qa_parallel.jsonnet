local params = import 'tgif_qa.jsonnet';

params + {
  model+: {
    text_video_mode: 'parallel',
  }
}
