local pretrained_model = 'bert-base-uncased';

(import 'tgif_qa.jsonnet') + {
  embedding_size: 768,

  dataset_reader+: {
    tokenizer: {
      type: 'word',
      word_splitter: 'bert-basic',
    },
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
      embedder_to_indexer_map: {
        bert: ['bert', 'bert-offsets']
      },
      token_embedders: {
        bert: {
          type: 'bert-pretrained',
          pretrained_model: pretrained_model,
          requires_grad: true,
        }
      }
    }
  }
}
