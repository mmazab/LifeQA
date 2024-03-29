(import 'lqa.libsonnet') + {
  embedding_size:: 300,
  video_channel_size:: 2048,
  encoder:: {
    type: 'lstm_patched',
    bidirectional: false,
    input_size: error 'Must override',
    hidden_size: 50,
    num_layers: 2,
    dropout: 0.2,
    return_all_layers: true,
    return_all_hidden_states: true,

    num_directions:: (if self.bidirectional then 2 else 1),
    output_size:: $.encoder.num_layers * $.encoder.num_directions * $.encoder.hidden_size,
  },

  dataset_reader+: {
    video_features_to_load: ['resnet-pool5'],
    join_question_and_answers: true,
    frame_step: 27,
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    },
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
    video_encoder: $.encoder + {
      input_size: $.video_channel_size
    },
    text_encoder: $.encoder + {
      input_size: $.embedding_size
    },
    classifier_feedforward: {
      input_dim: $.encoder.output_size,
      num_layers: 1,
      hidden_dims: [1],
      activations: ['linear'],
    },
    regularizer: [
      [
        'weight',
        {
          type: 'l2',
          alpha: 0.001,
        }
      ]
    ]
  },
  iterator: {
    sorting_keys: [['video_features', 'dimension_0']],
    batch_size: 8,
  },
  trainer+: {
    trainer: {
      num_epochs: 40,
      patience: 10,
      grad_clipping: 5.0,
      validation_metric: '+accuracy',
      optimizer: {
        type: 'adagrad'
      },
    }
  }
}
