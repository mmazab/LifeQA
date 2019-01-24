local config = import 'tgif_qa.jsonnet';
local pretrained_model = 'bert-base-uncased';

config + {
  embedding_size: 768,

  dataset_reader+: {
    token_indexers: {
      bert: {
        type: 'bert-pretrained',
        pretrained_model: pretrained_model,
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