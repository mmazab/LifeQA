local config = import 'tgif_qa.jsonnet';

config + {
  dataset_reader+: {
    lazy: true
  }
}
