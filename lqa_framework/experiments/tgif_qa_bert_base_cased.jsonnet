local params = import 'tgif_qa.jsonnet';
local pretrained_model = 'bert-base-cased';

params + {
  embedding_size: 768,

  dataset_reader+: {
    token_indexers: {
      bert: {
        type: 'bert-pretrained',
        pretrained_model: pretrained_model,
        do_lowercase: false,
      }
    }
  },
  model+: {
    text_field_embedder: {
      allow_unmatched_keys: true,
      token_embedders: {
        bert: {
          type: 'bert-pretrained',
          pretrained_model: pretrained_model,
        }
      }
    }
  }
}
