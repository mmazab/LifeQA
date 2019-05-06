local pretrained_model = 'bert-base-cased';

(import 'tgif_qa_bert_base_cased.jsonnet') + {
  model+: {
    text_field_embedder+: {
      token_embedders+: {
        bert+: {
          top_layer_only: true,
        }
      }
    }
  }
}
