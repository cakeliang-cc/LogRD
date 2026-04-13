import argparse

import pandas as pd
import torch
import random
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")

from bert_pytorch.predict_log import Predictor
from bert_pytorch.train_log import Trainer
import os

dirname = os.path.dirname(__file__)



def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "../output/bgl/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["hyper_model_path"] = options["model_dir"] + "hyper_best_bert.pth"
options["train_vocab"] = options["output_dir"] + "train"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"  # pickle file
options["window_size"] = 512
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 2
options["mask_ratio"] = 0.5
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.125
# 先改成少量的数据进行debug
options["test_ratio"] = 0.1
# features
options["is_logkey"] = True
options["is_time"] = False


options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 100
options["n_epochs_stop"] = 10
options["batch_size"] = 8

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = (1e-3)
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 120
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

options["train_vocab"] = options["output_dir"]+"train_vocab.csv"
options["vocab_size"] = len(pd.read_csv(options["train_vocab"])) + 1


# TODO 看情况修改
# test的数据集
options["normal_test"] = options["output_dir"]+"test_normal_processed.csv"
options["anomaly_test"] = options["output_dir"]+"df_abnormal_processed.csv"


# embedding_arr_path
# TODO 看情况修改
options["embedding_path"] = "../WhiteningNpy/whitened_embeddings_256d.npy"

options["similarity"] = 0.40
options["threshold1"] = 0.30

options["no_replace_path"] = options["output_dir"]+"no_similarity_whitened.csv"
options["ab_replace_path"] = options["output_dir"]+"ab_similarity_whitened.csv"

seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)


    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        print("-------------------------------begin to train--------------------------------------------")
        Trainer(options).train()
        print("-------------------------------train_success---------------------------------------------")

    elif args.mode == 'predict':
        print("predict")
        Predictor(options).predict()
        print("-------------------------------predict_success---------------------------------------------")







