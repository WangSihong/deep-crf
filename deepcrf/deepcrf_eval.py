
from .model import GraphDeepCRFModel
from .config import load_config_from_json_file
from .default_custom import DefaultTransform
import sys, getopt


def main():
    test_input_path = ""
    output_path = ""
    config_file_path = ""
    model_path = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:],"he:o:c:m:",["test=","output=", "config=","model="])
    except getopt.getopt.GetoptError:
        print('deepcrf_eval -e <test file> -c <config file> -m <model save path>')
        sys.exit(2)
        
    if len(opts) == 0:
        print('deepcrf_eval -e <test file> -c <config file> -m <model file path>')
        sys.exit()
        
    for opt, arg in opts:
        if opt == '-h':
            print('deepcrf_eval -e <test file> -c <config file> -m <model file path>')
            sys.exit()
        elif opt in ("-e", "--test"):
            test_input_path = arg
        elif opt in ("-o", "--output"):
            output_path = arg
        elif opt in ("-c", "--config"):
            config_file_path = arg
        elif opt in ("-m", "--model"):
            model_path = arg
        
    if test_input_path == "":
        print("Please input -e or --test to set test_input_path.")
        sys.exit()
         
    if output_path == "":
        print("Default output filename: eval-out.txt")
        output_path = "eval-out.txt"
        
    if config_file_path == "":
        print("Please input -c or --config to set config_file_path.")
        sys.exit()
            
    if model_path == "":
        print("Please input -m or --model to set model_path.")
        sys.exit()
     
    config = load_config_from_json_file(config_file_path)
    model = GraphDeepCRFModel(model_path, config, DefaultTransform)
    
    ofp = open(output_path, "w")
    with open(test_input_path, "r") as fp:
        rets = model.eval(fp)
        for ret in rets:
            for pred in ret:
                ptags = "\n".join([p[0] + "\t" + p[1] for p in pred])
                ofp.write(ptags + "\n\n")
    ofp.flush()
    ofp.close()
