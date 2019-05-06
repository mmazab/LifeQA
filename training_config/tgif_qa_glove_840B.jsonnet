(import 'tgif_qa.jsonnet') + {
  dataset_reader+: {
    token_indexers+: {
      tokens+: {
        lowercase_tokens: false
      }
    }
  },
  model+: {
    text_field_embedder+: {
      token_embedders+: {
        tokens+: {
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz'
        }
      }
    }
  }
}
