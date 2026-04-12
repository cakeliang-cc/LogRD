import argparse
import os
import sys

import pandas as pd
import torch
import random
import numpy as np

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bert_pytorch.predict_log import Predictor
from bert_pytorch.train_log import Trainer

dirname = os.path.dirname(__file__)



def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "../output/hdfs/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["hyper_model_path"] = options["model_dir"] + "hyper_best_bert.pth"
options["train_vocab"] = options["output_dir"] + "train"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"  # pickle file
options["window_size"] = 512
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10
options["mask_ratio"] = 0.5
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.125
# 先改成少量的数据进行debug
options["test_ratio"] = 1
# features
options["is_logkey"] = True
options["is_time"] = False


options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 768 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 100
options["n_epochs_stop"] = 10
options["batch_size"] = 32

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
options["num_candidates"] = 3
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

options["train_vocab"] = options["output_dir"] + "train_vocab.csv"
_train_vocab_abs = os.path.normpath(os.path.join(dirname, options["train_vocab"]))
if os.path.isfile(_train_vocab_abs):
    options["vocab_size"] = len(pd.read_csv(_train_vocab_abs)) + 1
else:
    options["vocab_size"] = 2


# TODO 看情况修改
# test的数据集
options["normal_test"] = options["output_dir"]+"test_normal.csv"
options["anomaly_test"] = options["output_dir"]+"testAnomaly.csv"


# embedding_arr_path
# TODO 看情况修改
options["embedding_path"] = "../WhiteningNpy/HDFS.npy"


options["threshold1"] = 0.50


seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(mode="train")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(mode="predict")
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)

    proc_parser = subparsers.add_parser(
        "process",
        help="从 raw HDFS.log + anomaly_label 生成 CSV、all_vocab、train_vocab 与 SimCSE 嵌入 HDFS.npy",
    )
    proc_parser.set_defaults(mode="process")
    _raw = os.path.normpath(os.path.join(dirname, "../output/hdfs/raw/HDFS.log"))
    _lab = os.path.normpath(os.path.join(dirname, "../output/hdfs/raw/anomaly_label.csv"))
    _out = os.path.normpath(os.path.join(dirname, "../output/hdfs/"))
    _npy_def = os.path.normpath(os.path.join(dirname, "../WhiteningNpy/HDFS.npy"))
    proc_parser.add_argument("--log", type=str, default=_raw, help="HDFS.log 路径")
    proc_parser.add_argument("--label", type=str, default=_lab, help="anomaly_label.csv")
    proc_parser.add_argument("--out_dir", type=str, default=_out, help="输出 train/test CSV 与 all_vocab 的目录")
    proc_parser.add_argument("--train_ratio", type=float, default=0.8, help="Normal 块中用于 train 的比例")
    proc_parser.add_argument("--seed", type=int, default=1234, help="split=random 时的随机种子")
    proc_parser.add_argument(
        "--split",
        type=str,
        choices=("sequential", "random"),
        default="sequential",
        help="Normal 块划分方式：sequential=按 BlockId 排序后切分；random=打乱后切分",
    )
    proc_parser.add_argument("--no-embed", action="store_true", help="不跑 SimCSE，不生成 HDFS.npy")
    proc_parser.add_argument(
        "--embed_model",
        type=str,
        default="princeton-nlp/sup-simcse-bert-base-uncased",
        help="HuggingFace 有监督 SimCSE 模型 id",
    )
    proc_parser.add_argument("--embed_batch_size", type=int, default=64)
    proc_parser.add_argument(
        "--whitening_npy",
        type=str,
        default=_npy_def,
        help="SimCSE 嵌入 .npy 输出路径（默认 WhiteningNpy/HDFS.npy）",
    )

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == "process":
        from hdfs_process import run_process

        run_process(
            log_path=args.log,
            label_path=args.label,
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            split=args.split,
            embed=not args.no_embed,
            embed_model=args.embed_model,
            embed_batch_size=args.embed_batch_size,
            whitening_npy=args.whitening_npy,
        )
    elif args.mode == 'train':
        print("-------------------------------begin to train--------------------------------------------")
        Trainer(options).train()
        print("-------------------------------train_success---------------------------------------------")

    elif args.mode == "predict":
        print("predict")
        Predictor(options).predict()
        print("-------------------------------predict_success---------------------------------------------")







