local params = import 'tgif_qa.jsonnet';

params + {
  model+: {
    loss: 'cross-entropy',
  }
}
