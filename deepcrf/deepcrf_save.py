# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
from .config import load_config_from_json_file
from . import utils
import sys, getopt

# Parameters
# ==================================================
def load_session(sess_file, config):
    graph = tf.Graph()
    graph.as_default()
    session_conf = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    sess.as_default()
    saver = tf.train.import_meta_graph("{}.meta".format(sess_file))
    saver.restore(sess, sess_file)
    return sess

def main():
    checkpoint_path = ""
    model_save_path = ""
    config_file_path = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hc:p:m:",["checkpoint=","model="])
    except getopt.getopt.GetoptError:
        print('deepcrf_learn -c <checkpoint path> -m <model file path>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('deepcrf_learn -c <checkpoint path> -m <model file path>')
            sys.exit()
        elif opt in ("-p", "--checkpoint"):
            checkpoint_path = arg
        elif opt in ("-m", "--model"):
            model_save_path = arg
        elif opt in ("-c", "--config"):
            config_file_path = arg
   
    if checkpoint_path == "":
        print("Please input -p or --checkpoint to set checkpoint_path.")
        sys.exit()
     
    if model_save_path == "":
        print("Please input -m or --model to set model_save_path.")
        sys.exit()
     
    if config_file_path == "":
        print("Please input -c or --config to set config_file_path.")
        sys.exit()
     
    config = load_config_from_json_file(config_file_path, "", "")
    
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    sess = load_session(checkpoint_file, config)
    
    scopes = []
    if utils.tf_version_uper_than("1.3.0"):
        scopes.append("viterbi/cond/Merge")
        scopes.append("viterbi/cond/Merge_1")
    else:
        scopes.append("crf/transitions")
    scopes.append("logits/logits")
    scopes.append("logits/prob")
    
    graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, scopes)
    with tf.gfile.GFile(model_save_path+".pb", "wb") as f:  
        f.write(graph.SerializeToString())

