local config = import 'tgif_qa.jsonnet';

config + {
  model+: {
    loss: 'cross-entropy',
  }
}
