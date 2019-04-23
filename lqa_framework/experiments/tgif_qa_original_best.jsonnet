// This experiment tries to stick to the best configuration of the original implementation of TGIF-QA.

local config = import 'tgif_qa.jsonnet';

config + {
  video_channel_size:: 2048 + 2048,

  dataset_reader+: {
    video_features_to_load: ['resnet-pool5', 'resof'],
    frame_step: 4,
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    }
  },
  vocabulary+: {
    max_vocab_size: 22852  // Ref: https://github.com/explosion/spaCy/issues/1341
  },
  model+: {
    // TODO: other dropouts?
    encoder+: {
      type: 'lstm_patched',
      bidirectional: false,
      hidden_size: 512,
      num_layers: 2,
      dropout: 0.2, // The original implementation uses dropout before and after every layer.
    },
    text_field_embedder+: {
      token_embedders+: {
        tokens+: {
          // We determined that TGIF-QA codebase uses spaCy 1.9.0, which uses en_core_web_sm 1.2.0, which uses 300d
          // GloVe embeddings from Common Crawl with a restricted vocabulary, I guess the uncased one. So the best idea
          // is to use the one that seems the same corpus (maybe the exact same embeddings) but restricting the
          // vocabulary.
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.42B.300d.txt.gz'
        }
      }
    },
    temporal_attention: {
      type: 'mlp',
      // Note: the original implementation takes the state for each layer, not just the last one.
      matrix_size: $.encoder.output_size / $.encoder.num_layers,
      vector_size: $.encoder.output_size,
    },
    classifier_feedforward: {  // TODO: check the hyperparams here
      input_dim: $.encoder.output_size,
      num_layers: 2,
      hidden_dims: [512],
      activations: ['tanh', 'linear'],
    },
  },
  iterator: {
    type: 'basic',
    batch_size: 64,
  },
  trainer: {
    num_epochs: 40,
    optimizer: {
      type: 'adam'
    },
  }
}
