local config = import 'tgif_qa.jsonnet';

config + {
  embedding_size: 100,

  model+: {
    text_field_embedder+: {
      token_embedders+: {
        tokens+: {
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz',
        }
      }
    }
  }
}
