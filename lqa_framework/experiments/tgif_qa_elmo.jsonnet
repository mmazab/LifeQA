local config = import 'tgif_qa.jsonnet';

config + {
  embedding_size: 1024,

  dataset_reader+: {
    token_indexers: {
      elmo: {
        type: 'elmo_characters'
      }
    }
  },
  model+: {
    text_field_embedder: {
      token_embedders: {
        elmo: {
          type: 'elmo_token_embedder',
          options_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',
          weight_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
          do_layer_norm: false,
          dropout: 0.5,
        }
      }
    }
  }
}
