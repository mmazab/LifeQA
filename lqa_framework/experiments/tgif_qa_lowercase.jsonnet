local params = import 'tgif_qa.jsonnet';

params + {
  dataset_reader+: {
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    }
  }
}
