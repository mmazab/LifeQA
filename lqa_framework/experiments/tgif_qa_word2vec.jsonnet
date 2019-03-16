local params = import 'tgif_qa.jsonnet';

params + {
  model+: {
    text_field_embedder+: {
      token_embedders+: {
        tokens+: {
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz'
        }
      }
    }
  }
}
