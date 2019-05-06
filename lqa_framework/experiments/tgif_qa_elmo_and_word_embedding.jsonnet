local original_config = import 'tgif_qa.jsonnet';
local params = import 'tgif_qa_elmo.jsonnet';

params + {
  word_embedding_size:: original_config.model.text_field_embedder.token_embedders.tokens.embedding_dim,
  embedding_size: super.embedding_size + self.word_embedding_size,

  dataset_reader+: {
    token_indexers+: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      }
    }
  },
  model+: {
    text_field_embedder+: {
      token_embedders+: original_config.model.text_field_embedder.token_embedders
    }
  }
}
