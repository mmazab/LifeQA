local params = import 'lqa.libsonnet';

params + {
  embedding_size:: 300,
  text_encoder:: {
    type: 'gru',
    bidirectional: false,
    input_size: $.embedding_size,
    hidden_size: 100,
    num_layers: 1,
    dropout: 0.2,
  },

  dataset_reader+: {
    video_features_to_load: 'resnet-pool5'
  },
  model: {
    type: 'tgif_qa',
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
    video_encoder: $.text_encoder + {
      input_size: 2048,
    },
    question_encoder: $.text_encoder,
    answers_encoder: $.text_encoder,
    classifier_feedforward: {
      input_dim: $.text_encoder.hidden_size * $.text_encoder.num_layers,
      num_layers: 1,
      hidden_dims: [1],
      activations: ['linear'],
    }
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['question', 'num_tokens']],  # TODO: How to put video_features here?
    batch_size: 64,
  },
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
