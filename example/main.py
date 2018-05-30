# -*- coding=utf-8 -*-

from custom import CustomTransform
from deepcrf import load_config_from_json_file
from deepcrf import train

train_input_path = "data/train.txt"
test_input_path = "data/test.txt"
config = load_config_from_json_file("config.json", train_input_path, test_input_path)

model = train(config, CustomTransform, True, True)
model.save("your_model_file_name")

with open("data/eval.txt") as fp:
    model.eval(fp)
