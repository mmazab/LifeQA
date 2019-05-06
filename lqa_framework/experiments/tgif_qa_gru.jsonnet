local config = import 'tgif_qa.jsonnet';

config + {
  encoder+: {
    type: 'gru_patched'
  }
}
