import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from Net.Net import BERTLog, BERT
from .optim_schedule import ScheduledOptim


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.
    """

    def __init__(self, vocab_size, bert: BERT, out_size: int,
                 train_dataloader: DataLoader, embedding_arr, valid_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, is_logkey=True, is_time=False,
                 run_weight_alpha: float = 0.3,
                 ):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param valid_dataloader: valid dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        self.vocab_size = vocab_size
        self.embedding_arr = embedding_arr
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLog(bert, out_size, self.vocab_size).to(self.device)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optim = None
        self.optim_schedule = None
        self.init_optimizer()

        self.cls_criterion = nn.NLLLoss(ignore_index=0)
        self.criterion = nn.MSELoss()

        self.time_criterion = nn.MSELoss()


        self.objective = None # self.objective = None: 这行代码设置了 deep SVDD 模型中的超参数 objective 的初始值为 None。

        self.log_freq = log_freq

        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.is_logkey = is_logkey
        self.is_time = is_time
        self.run_weight_alpha = run_weight_alpha  # run 中部的权重（应该 < 1）

    def manual_nll_loss(self, log_probs, target, original_sequence, ignore_index=0, alpha=0.3):
        """
        手动实现的负对数似然损失函数 (Negative Log Likelihood Loss)
        支持基于 run 的权重计算
        
        NLLLoss 的计算公式：
        loss = -log(P(y_true | x))
        其中 P(y_true | x) 是模型预测的真实类别 y_true 的概率
        权重规则：
        - 如果 token_i 在 run 起点或转移位置（新 run 的起点）：w(i) = 1
        - 如果 token_i 在 run 终点或中部：w(i) = alpha(run_len)
          其中 alpha(run_len) = max(0.1, 0.5 / log(1 + run_len))
        
        Args:
            log_probs: 对数概率，形状为 (batch_size, num_classes, seq_len)
                      注意：输入应该是 LogSoftmax 的输出，已经是 log 空间
            target: 真实标签，形状为 (batch_size, seq_len)，包含类别索引
            original_sequence: 原始序列，形状为 (batch_size, seq_len)，用于计算 run 权重
            ignore_index: 要忽略的索引（通常是填充位置的索引，默认为0）
            alpha: 兼容旧接口（当前实际按 run_len 动态计算，公式如上）
        
        Returns:
            loss: 标量张量，所有有效位置的加权平均负对数似然损失
        """
        # log_probs 形状: (batch_size, num_classes, seq_len)
        # target 形状: (batch_size, seq_len)
        # original_sequence 形状: (batch_size, seq_len)
        batch_size, num_classes, seq_len = log_probs.shape
        
        # 创建 mask，标记哪些位置需要计算损失（非 ignore_index 的位置）
        # mask 形状: (batch_size, seq_len)，True 表示需要计算，False 表示忽略
        mask = (target != ignore_index)
        
        # 计算 run 权重
        # weights 形状: (batch_size, seq_len)
        weights = self._compute_run_weights(original_sequence, mask, alpha)
        
        # 使用 gather 从 log_probs 中提取对应类别索引的对数概率
        # target 需要扩展维度以匹配 gather 的要求
        # target_expanded 形状: (batch_size, 1, seq_len)
        target_expanded = target.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # 使用 gather 在类别维度上收集对应索引的对数概率
        # gather(dim=1, index=target_expanded) 从每个位置选择对应类别的 log_prob
        # 结果形状: (batch_size, 1, seq_len)
        selected_log_probs = log_probs.gather(dim=1, index=target_expanded)
        
        # 去掉中间的维度，得到 (batch_size, seq_len)
        selected_log_probs = selected_log_probs.squeeze(1)  # (batch_size, seq_len)
        
        # 取负值得到负对数似然（因为输入已经是 log 概率，所以直接取负）
        # nll 形状: (batch_size, seq_len)
        nll = -selected_log_probs
        
        # 应用权重和 mask
        # 只对有效位置（mask=True）计算损失，并应用 run 权重
        nll_weighted = nll * weights * mask.float()  # 无效位置变为 0
        
        # 计算加权后的有效位置总数（用于归一化）
        total_weight = (weights * mask.float()).sum()
        
        # 避免除零错误
        if total_weight > 0:
            # 对所有有效位置求加权平均
            loss = nll_weighted.sum() / total_weight
        else:
            # 如果没有有效位置，返回 0
            loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
        
        return loss
    
    def _compute_run_weights(self, original_sequence, mask, alpha):
        """
        根据原始序列计算每个位置的 run 权重
        
        权重规则：
        - run 第一个位置权重 1
        - run 的转移点（新 run 的起点）和 run 内其它位置：w(i) = alpha(run_len)
          其中 alpha(run_len) = max(0.1, 0.5 / log(1 + run_len))
        
        Args:
            original_sequence: 原始序列，形状为 (batch_size, seq_len)
            mask: 有效位置 mask，形状为 (batch_size, seq_len)
            alpha: 兼容旧接口（当前实际按 run_len 动态计算，公式如上）
        
        Returns:
            weights: 权重张量，形状为 (batch_size, seq_len)
        """
        batch_size, seq_len = original_sequence.shape
        device = original_sequence.device
        
        # 初始化为最小权重 0.1，后面按 run 长度覆盖
        weights = torch.full((batch_size, seq_len), 0.1, device=device, dtype=torch.float)
        
        # 对每个样本计算 run 权重
        for b in range(batch_size):
            seq = original_sequence[b]  # (seq_len,)
            valid_mask = mask[b]  # (seq_len,)
            
            # 只处理有效位置（非填充位置）
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
            
            # 获取有效位置的序列值
            valid_seq = seq[valid_indices]
            
            # 扫描 run：连续相同值为一个 run
            run_start = 0
            while run_start < len(valid_indices):
                run_value = valid_seq[run_start].item()
                run_end = run_start
                # 找到当前 run 的结束位置
                while run_end + 1 < len(valid_indices) and valid_seq[run_end + 1].item() == run_value:
                    run_end += 1

                run_len = run_end - run_start + 1
                # 按 run 长度计算 alpha(run_len)
                alpha_run = max(0.1, 0.5 / math.log(1 + run_len))

                # 设权重：
                # - 第一个 run 的起点权重 1
                # - 后续 run 的起点（转移点）和 run 内其它位置使用 alpha(run_len)
                for i in range(run_start, run_end + 1):
                    idx_val = valid_indices[i].item()
                    if i == run_start and run_start == 0:
                        weights[b, idx_val] = 1.0
                    else:
                        weights[b, idx_val] = alpha_run

                run_start = run_end + 1
        
        return weights

    def init_optimizer(self):

        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=self.warmup_steps)

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self.train_data, start_train=True)

    def valid_train(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.train_data, start_train=False)

    def valid(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.valid_data, start_train=False)
    #
    def iteration(self, epoch, data_loader, start_train):
        """
        loop over the data_loader for training or validing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or valid
        :return: None
        """
        str_code = "train" if start_train else "valid"

        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)
        total_loss = 0.0
        total_cls_loss = 0.0

        for i, data in data_iter:
            data[0] = data[0].to(self.device)
            data[1] = data[1].to(self.device)
            data[2] = data[2].to(self.device)
            data[3] = data[3].to(self.device)
            # data[4] 是原始序列，用于后续计算 run 权重（暂时不使用）
            data[4] = data[4].to(self.device)

            result = self.model.forward(data[0], data[3])

            cls_lm_output = result["cls_output"]

            # 使用手动实现的 NLLLoss（支持 run 权重）
            cls_loss = torch.tensor(0) if not self.is_logkey else self.manual_nll_loss(
                cls_lm_output.transpose(1, 2), data[2], data[4], ignore_index=0, alpha=self.run_weight_alpha
            )
            
            # 如果想使用 PyTorch 内置的 NLLLoss，可以替换为：
            # cls_loss = torch.tensor(0) if not self.is_logkey else self.cls_criterion(cls_lm_output.transpose(1, 2), data[2])
            total_cls_loss += cls_loss.item()
            loss = cls_loss

            total_loss += loss.item()

            # 3. backward and optimization only in train
            if start_train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
        avg_loss = total_loss / totol_length
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, loss={}".format(epoch, str_code, avg_loss))
        print(f"cls loss: {total_cls_loss / totol_length}\n")

        return avg_loss



    #
    def save_log(self, save_dir, surfix_log):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(save_dir + key + f"_{surfix_log}.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")
    #
    def save(self, save_dir="output/bert_trained.pth"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        torch.save(self.model, save_dir)
        # self.bert.to(self.device)
        print(" Model Saved on:", save_dir)
        return save_dir


