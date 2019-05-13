(import 'lqa.libsonnet') + {
  embedding_size:: 100 + 100,

  dataset_reader+: {
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
      token_characters: {
        type: 'characters',
        character_tokenizer: {
          byte_encoding: 'utf-8',
          start_tokens: [259],
          end_tokens: [260]
        },
        min_padding_length: 5
      }
    }
  },
  model: {
    text_encoder:: {
      type: 'lstm',
      bidirectional: true,
      input_size: $.embedding_size,
      hidden_size: 100,
      num_layers: 1,
    },

    type: 'bidaf_lqa',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz',
          embedding_dim: 100,
          trainable: false,
        },
        token_characters: {
          type: 'character_encoding',
          embedding: {
            num_embeddings: 262,
            embedding_dim: 16
          },
          encoder: {
            type: 'cnn',
            embedding_dim: 16,
            num_filters: 100,
            ngram_filter_sizes: [5]
          },
          dropout: 0.2
        }
      }
    },
    question_encoder: self.text_encoder,
    captions_encoder: self.text_encoder,
    answers_encoder: self.text_encoder,
    phrase_layer: {
      type: 'lstm',
      bidirectional: true,
      input_size: $.embedding_size,
      hidden_size: 100,
      num_layers: 1,
    },
    modeling_layer: {
      type: 'lstm',
      bidirectional: true,
      input_size: 800,
      hidden_size: 100,
      num_layers: 1,
    },
    num_highway_layers: 2,
    classifier_feedforward: {
      input_dim: 200,
      num_layers: 2,
      hidden_dims: [200, 200],
      activations: ['relu', 'linear'],
      dropout: [0.2, 0.0],
    },
    classifier_feedforward_answers: {
      input_dim: 800,
      num_layers: 2,
      hidden_dims: [200, 200],
      activations: ['relu', 'linear'],
      dropout: [0.2, 0.0],
    }
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['captions', 'num_fields'], ['question', 'num_tokens']],
    batch_size: 64,
  },
  trainer+: {
    trainer: {
      num_epochs: 40,
      patience: 10,
      grad_clipping: 5.0,
      validation_metric: '+accuracy',
      optimizer: {
        type: 'adagrad',
      },
    }
  }
}
