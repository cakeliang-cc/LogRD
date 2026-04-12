import gc
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from Net.Net import BERT, BERTLog
from bert_pytorch.dataset.log_dataset import LogDataset
from bert_pytorch.dataset.sample import generate_train_valid
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.trainer.optim_schedule import ScheduledOptim


def save(model, save_dir="output/bert_trained.pth"):
    """
    Saving the current BERT model on file_path

    :param epoch: current epoch number
    :param file_path: model output path which gonna be file_path+"ep%d" % epoch
    :return: final_output_path
    """
    torch.save(model, save_dir)
    # self.bert.to(self.device)
    print(" Model Saved on:", save_dir)
    return save_dir

def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))

class Trainer():
    def __init__(self, options):
        self.vocab_size = options["vocab_size"]
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.output_path = options["output_dir"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.sample_ratio = options["train_ratio"]
        self.valid_ratio = options["valid_ratio"]
        self.seq_len = options["seq_len"]
        self.max_len = options["max_len"]
        self.corpus_lines = options["corpus_lines"]
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_weight_decay = options["adam_weight_decay"]
        self.with_cuda = options["with_cuda"]
        self.cuda_devices = options["cuda_devices"]
        self.log_freq = options["log_freq"]
        self.epochs = options["epochs"]
        self.hidden = options["hidden"]
        self.layers = options["layers"]
        self.attn_heads = options["attn_heads"]
        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale = options["scale"]
        self.scale_path = options["scale_path"]
        self.n_epochs_stop = options["n_epochs_stop"]
        self.mask_ratio = options["mask_ratio"]
        self.min_len = options['min_len']
        # 嵌入数组
        self.embedding_path = options["embedding_path"]
        self.embedding_arr = np.load(self.embedding_path)

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")
        logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(self.output_path + "train.csv",
                                                                                  window_size=self.window_size,
                                                                                  adaptive_window=self.adaptive_window,
                                                                                  valid_size=self.valid_ratio,
                                                                                  sample_ratio=self.sample_ratio,
                                                                                  scale=self.scale,
                                                                                  scale_path=self.scale_path,
                                                                                  seq_len=self.seq_len,
                                                                                  min_len=self.min_len
                                                                                  )
        self.logkey_train = logkey_train
        self.time_train = time_train

        self.logkey_valid = logkey_valid
        self.time_valid = time_valid
        self.vocab = options["train_vocab"]


    def train(self):
        print("train")
        print("\nLoading Train Dataset")
        # 接在训练数据集和验证数据集
        print("-" * 10)
        print('window_size=' + str(self.window_size))  # 这个好像暂时没什么用处
        print('adaptive_window:' + str(self.adaptive_window))
        print('self.valid_ratio:' + str(self.valid_ratio))
        print('sample_ratio=' + str(self.sample_ratio))
        print('scale=' + str(self.scale))
        print('scale_path=' + str(self.scale_path))
        print('seq_len=' + str(self.seq_len))  # 序列的最长长度
        print('min_len=' + str(self.min_len))  # 序列的最短长度
        print("-" * 10)


        
        train_dataset = LogDataset(self.vocab, self.logkey_train, self.time_train, self.embedding_arr, seq_len=self.seq_len,
                                   corpus_lines=self.corpus_lines, on_memory=self.on_memory, mask_ratio=self.mask_ratio)
        print("\nLoading valid Dataset")
        valid_dataset = LogDataset(self.vocab, self.logkey_valid, self.time_valid, self.embedding_arr, seq_len=self.seq_len, on_memory=self.on_memory,
                                   mask_ratio=self.mask_ratio)
        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        del train_dataset
        del valid_dataset

        gc.collect()
        print("Building BERT model")
        # (self, max_len=512, emded_size=768, hidden=768, n_layers=12, attn_heads=12, dropout=0.1)

        bert = BERT(max_len=self.max_len, emded_size=self.hidden ,hidden=self.hidden, n_layers=self.layers,
                    attn_heads=self.attn_heads)
        print("Creating BERT Trainer")

        self.trainer = BERTTrainer(self.vocab_size, bert, self.hidden, train_dataloader=self.train_data_loader,
                                   embedding_arr= self.embedding_arr,
                                   valid_dataloader=self.valid_data_loader,
                                   lr=self.lr, betas=(self.adam_beta1, self.adam_beta2),
                                   weight_decay=self.adam_weight_decay,
                                   with_cuda=self.with_cuda, cuda_devices=self.cuda_devices, log_freq=self.log_freq,
                                   is_logkey=self.is_logkey, is_time=self.is_time,)
        self.start_iteration(surfix_log="log2")
        self.plot_train_valid_loss("_log2")



    def start_iteration(self, surfix_log):
        print("Training Start")
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.epochs):

            self.trainer.train_data = self.train_data_loader
            self.trainer.valid_data = self.valid_data_loader
            print("\n")

            _ = self.trainer.train(epoch)
            avg_loss = self.trainer.valid(epoch)
            self.trainer.save_log(self.model_dir, surfix_log)
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                print("Early stopping")
                break

    def plot_train_valid_loss(self, surfix_log):
        train_loss = pd.read_csv(self.model_dir + f"train{surfix_log}.csv")
        valid_loss = pd.read_csv(self.model_dir + f"valid{surfix_log}.csv")
        sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
        sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
        plt.title("epoch vs train loss vs valid loss")
        plt.legend()
        plt.savefig(self.model_dir + "train_valid_loss.png")
        plt.show()
        print("plot done")


