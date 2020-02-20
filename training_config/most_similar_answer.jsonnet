local pretrained_file = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz';
local embedding_dim = 300;
local text_encoder = {
  type: 'bag_of_embeddings',
  embedding_dim: embedding_dim,
};

(import 'simple.libsonnet') + {
  dataset_reader+: {
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    }
  },
  vocabulary: {
    pretrained_files: {
      'tokens': pretrained_file
    },
    only_include_pretrained_words: true,
  },
  model: {
    type: 'most_similar_answer',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          pretrained_file: pretrained_file,
          embedding_dim: embedding_dim,
          trainable: false,
        }
      }
    },
    question_encoder: text_encoder,
    answers_encoder: text_encoder,
  }
}
