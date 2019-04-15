local params = import 'lqa.libsonnet';

params + {
  embedding_size:: 300,
  encoder:: {
    type: 'lstm_patched',
    bidirectional: true,
    input_size: $.embedding_size,
    hidden_size: 50,
    num_layers: 2,
    dropout: 0.2,
    return_all_layers: true,
    return_all_hidden_states: true,

    num_directions:: (if self.bidirectional then 2 else 1),
  },

  dataset_reader+: {
    video_features_to_load: ['resnet-pool5'],
    join_question_and_answers: true,
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
      input_size: 2048
    },
    text_encoder: $.encoder,
    classifier_feedforward: {
      input_dim: $.encoder.num_layers * $.encoder.num_directions * $.encoder.hidden_size,
      num_layers: 1,
      hidden_dims: [1],
      activations: ['linear'],
    }
  },
  iterator: {
    type: 'basic',  //bucket
    //sorting_keys: [['question_and_answers', 'num_tokens']],  # TODO: How to put video_features here?
    batch_size: 64,
  },
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
