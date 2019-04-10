// This experiment tries to stick to the best configuration of the original implementation of TGIF-QA.

local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader.video_features_to_load: ['resnet-pool5', 'c3d-fc6'],
  vocabulary.max_vocab_size: 22852,  // See https://github.com/explosion/spaCy/issues/1341)
  model+: {
    text_encoder+: {
      type: 'lstm',
      bidirectional: true,
      hidden_size: 256,
      num_layers: 2,
      dropout: [0.2, 0.2],
    },
    // I determined that TGIF-QA codebase uses spaCy 1.9.0, which uses en_core_web_sm 1.2.0, which uses 300d GloVe
    // embeddings from Common Crawl with a restricted vocabulary, I guess the uncased one. So the best idea is to use
    // the one that seems the same corpus (maybe the exact same embeddings) but restricting the vocabulary.
    text_field_embedder.token_embedders.tokens.pretrained_file:
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.42B.300d.txt.gz',
    video_encoder.input_size: 2048 + 4096,
  },
  iterator+: {
    batch_size: 32
  }
  trainer: {
    num_epochs: 40,
    patience: 10,
    optimizer: {
      type: 'adam'
    },
  }
}
