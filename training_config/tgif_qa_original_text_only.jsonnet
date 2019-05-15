// This experiment tries to stick to the best configuration of the original implementation of TGIF-QA, from the CVPR paper.

(import 'tgif_qa.jsonnet') + {
  dataset_reader+: {
    video_features_to_load: null,
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
    // TODO: other dropouts?,
    text_video_mode: 'text',
    encoder+:: {
      type: 'lstm_patched',
      bidirectional: false,
      hidden_size: 512,
      num_layers: 2,
      // The original implementation uses dropout before and after every layer, but this can only specify in between
      // layers.
      dropout: 0.2,
    },
    video_encoder: null,
    text_field_embedder+: {
      token_embedders+: {
        tokens+: {
          // We determined that TGIF-QA codebase uses spaCy 1.9.0, which uses en_core_web_sm 1.2.0, which uses 300d
          // GloVe embeddings from Common Crawl with a restricted vocabulary, I guess the uncased one. So the best idea
          // is to use the one that seems the same corpus (maybe the exact same embeddings) but restricting the
          // vocabulary.
          pretrained_file: 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
        }
      }
    },
  },
  iterator: {
    type: 'basic',
    batch_size: 64,  // The original repo uses 64 for all GPUs.
  },
  trainer+: {
    trainer: {
      num_epochs: 40,
      validation_metric: '+accuracy',
      optimizer: {
        type: 'adam'
      },
    }
  }
}
