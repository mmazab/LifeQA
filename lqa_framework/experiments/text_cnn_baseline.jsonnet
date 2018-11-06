local embedding_size = 300;

{
  "dataset_reader": {
	"type": "lqa"
  },
  "train_data_path": "data/lqa_train.json",
  "validation_data_path": "data/lqa_dev.json",
  "test_data_path": "data/lqa_test.json",
  "model": {
	"type": "text_baseline",
	"text_field_embedder": {
	  "tokens": {
		"type": "embedding",
		"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
		"embedding_dim": embedding_size,
		"trainable": false
	  }
	},
	"question_encoder": {
	  "type": "cnn",
	  "num_filters": 100,
	  "ngram_filter_sizes": [2, 3],
      "embedding_dim": embedding_size
	},
	"captions_encoder": {
	  "type": "cnn",
	  "num_filters": 100,
	  "ngram_filter_sizes": [2, 3],
      "embedding_dim": embedding_size
	},
	"answers_encoder": {
	  "type": "cnn",
	  "num_filters": 100,
	  "ngram_filter_sizes": [2, 3],
      "embedding_dim": embedding_size
	},
	"classifier_feedforward": {
	  "input_dim": 400,
	  "num_layers": 2,
	  "hidden_dims": [200, 200],
	  "activations": ["relu", "linear"],
	  "dropout": [0.2, 0.0]
	}
  },
  "iterator": {
	"type": "bucket",
	"sorting_keys": [["captions", "num_fields"], ["question", "num_tokens"]],
	"batch_size": 64
  },
  "trainer": {
	"num_epochs": 40,
	"patience": 10,
	"cuda_device": 0,
	"grad_clipping": 5.0,
	"validation_metric": "+accuracy",
	"optimizer": {
	  "type": "adagrad"
	}
  }
}
