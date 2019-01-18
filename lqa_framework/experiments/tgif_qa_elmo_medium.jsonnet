local config = import 'tgif_qa_elmo.jsonnet';

config + {
  embedding_size: 512,

  model+: {
    text_field_embedder+: {
      token_embedders+: {
        elmo+: {
          options_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json',
          weight_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5',
        }
      }
    }
  }
}
