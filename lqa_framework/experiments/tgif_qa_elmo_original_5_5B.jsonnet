local config = import 'tgif_qa_elmo.jsonnet';

config + {
  model+: {
    text_field_embedder+: {
      token_embedders+: {
        elmo+: {
          options_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
          weight_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
        }
      }
    }
  }
}
