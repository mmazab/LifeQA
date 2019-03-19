local params = import 'lqa.libsonnet';

params + {
  iterator: {
    type: 'basic'
  },
  trainer: {
    type: 'no_op'
  }
}
