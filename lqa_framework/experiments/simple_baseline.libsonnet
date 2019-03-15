local params = import 'lqa.libsonnet';

params + {
  iterator: {  # dummy
    type: 'basic',
  },
  trainer: {
    optimizer: {},  # dummy
  }
}
