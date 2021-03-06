(import 'lqa.libsonnet') + {
  embedding_size:: 300,

  dataset_reader+: {
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    },
    unroll_captions: false,
  },
  model: {
    text_encoder:: error 'Must override',

    type: 'text_memn2n',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz',
          embedding_dim: $.embedding_size,
          trainable: false,
        }
      }
    },
    question_encoder: self.text_encoder,
    captions_encoder: self.text_encoder,
    answers_encoder: self.text_encoder,
    projection_layer: {	
      input_dim: $.embedding_size,
      num_layers: 1,
      hidden_dims: [$.embedding_size],
      activations: ['relu'],
      dropout: [0.2],
    },
    classifier_feedforward: {
      input_dim: 400,
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
