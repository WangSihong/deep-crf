# -*- coding=utf-8 -*-
import json
import multiprocessing

class Config(object):
    def __init__(self, json_str):
        self._config_object = json.loads(json_str)

    def show(self):
        for attr, value in sorted(self.__dict__.items()):
            if attr == "_config_object":
                continue
            print("{}={}".format(attr.upper(), value))
        print("")


class DeepCRFConfig(Config):
    def __init__(self, json_str, train_input_path, test_input_path=None):
        super(DeepCRFConfig, self).__init__(json_str)
        self.train_input_path = train_input_path
        self.test_input_path = test_input_path

        self.lr = self._config_object["lr"]
        self.epoch = self._config_object["epoch"]

        self.num_tags = self._config_object.get("num_tags", 4)
        self.batch_size = self._config_object.get("batch_size", 64)
        self.evaluate_every = self._config_object.get("evaluate_every", 1000)
        self.checkpoint_every = self._config_object.get("checkpoint_every", 1000)
        self.num_checkpoints = self._config_object.get("num_checkpoints", 5)
        self.embedding_size = self._config_object.get("embedding_size", 100)

        self.input_keep_prob = self._config_object.get("input_keep_prob", 0.8)
        self.output_keep_prob = self._config_object.get("output_keep_prob", 0.5)

        self.hidden_size = self._config_object.get("hidden_size", 20)
        self.hidden_layer_num = self._config_object.get("hidden_layer_num", 1)

        self.allow_soft_placement = self._config_object.get("allow_soft_placement", True)
        self.log_device_placement = self._config_object.get("log_device_placement", False)

        # config for word2vec
        self.alpha = self._config_object.get("alpha", 0.025)
        self.window = self._config_object.get("window", 5)
        self.min_count = self._config_object.get("min_count", 3)
        self.sample = self._config_object.get("sample", 0.001)
        self.workers = self._config_object.get("workers", multiprocessing.cpu_count())
        self.negative = self._config_object.get("negative", 5)
        self.iter_times = self._config_object.get("iter_times", 5)

        self.show()


def load_config_from_json_file(path, train_input_path=None, test_input_path=None):
    with open(path, "r") as fp:
        json_str = fp.read()
        return DeepCRFConfig(json_str, train_input_path, test_input_path)