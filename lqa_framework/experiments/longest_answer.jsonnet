local params = import 'simple_baseline.libsonnet';

params + {
  model: {
    type: 'longest_answer'
  }
}
