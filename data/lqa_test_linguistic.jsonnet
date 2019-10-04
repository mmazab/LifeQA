std.mapWithKey(
  function(v_id, v) v + {questions: std.filter(function(q) q.answer_type != 'V', v.questions)},
  import 'lqa_test.json'
)
