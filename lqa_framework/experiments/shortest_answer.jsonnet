local params = import 'simple_baseline.libsonnet';

params + {
  model: {
    type: 'shortest_answer'
  }
}
