
from deepcrf import GraphDeepCRFModel
from deepcrf import load_config_from_json_file
from custom import CustomTransform

config = load_config_from_json_file("config.json")
model = GraphDeepCRFModel("your_model_path", config, CustomTransform)
ret = model.predict(["我","爱","北","京","天","安","门"])
print(ret)
