# -*- coding=utf-8 -*-

from .config import load_config_from_json_file
from .default_custom import DefaultTransform
from .train import train
import sys, getopt


def main():
    
    train_input_path = ""
    test_input_path = ""
    config_file_path = ""
    model_save_path = ""
    need_transform = True
    need_build_word2vec = True
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:e:c:m:f:w:",["train=","test=","config=","model=","transform=","word2vec"])
    except getopt.getopt.GetoptError:
        print('deepcrf_learn -t <trainfile> -e <testfile> -c <configfile> -m <model save path> [-f <do transform> -w <build word2vec>]')
        sys.exit(2)
        
    if len(opts) == 0:
        print('deepcrf_learn -t <trainfile> -e <testfile> -c <configfile> -m <model save path> [-f <do transform> -w <build word2vec>]')
        sys.exit()
        
    for opt, arg in opts:
        if opt == '-h':
            print('deepcrf_learn -t <trainfile> -e <testfile> -c <configfile> -m <model save path> [-f <do transform> -w <build word2vec>]')
            sys.exit()
        elif opt in ("-t", "--train"):
            train_input_path = arg
        elif opt in ("-e", "--test"):
            test_input_path = arg
        elif opt in ("-c", "--config"):
            config_file_path = arg
        elif opt in ("-m", "--model"):
            model_save_path = arg
        elif opt in ("-f", "--transform"):
            if arg == "true" or arg == "True" or arg == "1":
                need_transform = True
            elif arg == "false" or arg == "False" or arg == "0":
                need_transform = False
            else:
                print('The need transform arg must be [true|false|True|False|1|0]')
                sys.exit()
        elif opt in ("-w", "--word2vec"):
            if arg == "true" or arg == "True" or arg == "1":
                need_build_word2vec = True
            elif arg == "false" or arg == "False" or arg == "0":
                need_build_word2vec = False
            else:
                print('The need build word2vec arg must be [true|false|True|False|1|0]')
                sys.exit()
        
    if train_input_path == "":
        print("Please input -t or --train to set train_input_path.")
        sys.exit()
         
    if test_input_path == "":
        print("Please input -e or --test to set test_input_path.")
        sys.exit()
        
    if config_file_path == "":
        print("Please input -c or --config to set config_file_path.")
        sys.exit()
            
    if model_save_path == "":
        print("Please input -m or --model to set model_save_path.")
        sys.exit()
            
    config = load_config_from_json_file(config_file_path, train_input_path, test_input_path)

    model = train(config, DefaultTransform, need_transform, need_build_word2vec)
    model.save(model_save_path)