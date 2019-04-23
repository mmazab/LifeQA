local params = import 'tgif_qa_elmo.jsonnet';

params + {
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
