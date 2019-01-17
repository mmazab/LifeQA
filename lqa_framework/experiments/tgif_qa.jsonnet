local embedding_size = 300;
local rnn_type = "gru";
local rnn_hidden_size = 100;
local rnn_num_layers = 1;
local bidirectional = false;
local rnn_dropout = 0.2;
local feed_forward_hidden_size = rnn_hidden_size * rnn_num_layers;

{
  "dataset_reader": {
    "type": "lqa",
    "load_video_features": true
  },
  "train_data_path": "data/lqa_train.json",
  "validation_data_path": "data/lqa_dev.json",
  "test_data_path": "data/lqa_test.json",
  "model": {
    "type": "tgif_qa",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
          "embedding_dim": embedding_size,
          "trainable": false
        }
      }
    },
    "video_encoder": {
      "type": rnn_type,
      "bidirectional": bidirectional,
      "input_size": 2048,
      "hidden_size": rnn_hidden_size,
      "num_layers": rnn_num_layers,
      "dropout": rnn_dropout
    },
    "question_encoder": {
      "type": rnn_type,
      "bidirectional": bidirectional,
      "input_size": embedding_size,
      "hidden_size": rnn_hidden_size,
      "num_layers": rnn_num_layers,
      "dropout": rnn_dropout
    },
    "answers_encoder": {
      "type": rnn_type,
      "bidirectional": bidirectional,
      "input_size": embedding_size,
      "hidden_size": rnn_hidden_size,
      "num_layers": rnn_num_layers,
      "dropout": rnn_dropout
    },
    "classifier_feedforward": {
      "input_dim": feed_forward_hidden_size,
      "num_layers": 1,
      "hidden_dims": [1],
      "activations": ["linear"]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["question", "num_tokens"]],  # TODO: How to put video_features here?
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
