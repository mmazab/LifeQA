(import 'text_only_mem.libsonnet') + {
 dataset_reader+: {
    load_objects: true,
    combine_objects_n_frames: 50,
    top_k_objects: 5,
  },
  model+: {
    type: 'multimodal_memn2n',
    text_encoder:: {
      type: 'lstm',
      bidirectional: true,
      input_size: $.embedding_size,
      hidden_size: 100,
      num_layers: 1,
    }
  }
}
