(import 'tgif_qa_elmo.jsonnet') + {
  model+: {
    text_field_embedder+: {
      token_embedders+: {
        elmo+: {
          requires_grad: true
        }
      }
    }
  }
}
