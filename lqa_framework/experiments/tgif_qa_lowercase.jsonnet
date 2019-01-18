local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    }
  }
}
