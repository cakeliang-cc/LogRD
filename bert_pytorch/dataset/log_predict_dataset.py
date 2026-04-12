import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
# 这里实现的dataSet采用一个固定掩码的方式

def pad_lists_to_length(lists, seq_len, padding_value=0):
    padded_lists = []
    for lst in lists:
        if len(lst) < seq_len:
            padded_lst = lst + [padding_value] * (seq_len - len(lst))
        else:
            padded_lst = lst[:seq_len]
        padded_lists.append(padded_lst)
    return padded_lists

# 修改LogDataset使其适配我的任务
class LogDatasetTest(Dataset):
    def __init__(self, vocab, log_corpus, time_corpus,  embedding_arr, seq_len, corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15, flag = "odd"):
        """
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions 表示日志会话的数量。它指定了语料库中包含多少个日志会话
        :param encoding: 它用于指定在处理文本数据时应该使用的编码方式，例如 UTF-8、ASCII 等。
        :param on_memory:
        :param predict_mode: if predict
        """
        # vocab基本没什么用的

        self.seq_len = seq_len
        self.on_memory = on_memory
        self.encoding = encoding
        self.predict_mode = predict_mode
        self.log_corpus = log_corpus     # 实际上就是需要训练的数据
        self.time_corpus = time_corpus
        self.corpus_lines = len(log_corpus)
        self.mask_ratio = mask_ratio
        # 新增加的
        self.pad_index = 0
        self.mask_index = -1
        self.cls_index = -2
        self.embedding_arr = embedding_arr
        self.mask_embedding = np.mean(self.embedding_arr, axis=0)
        self.dim = self.embedding_arr.shape[1]
        # 用来限制当前是在奇数位置进行掩码还是在偶数位置进行掩码
        self.flag = flag
        self.vocab = vocab
        # 使用pandas的to_dict方法将dataframe转换为字典
        self.vocab_dict = dict(zip(self.vocab['contentId'], self.vocab['vocabId']))
        self.vocab_size = len(self.vocab) + 1

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        k_list = list(self.log_corpus[idx])
        k_list_int = [int(item) - 1 for item in k_list]
        selected_rows = np.array([self.embedding_arr[index] for index in k_list_int])
        # selected_rows = self.embedding_arr[k_list]
        cls_embedding = np.mean(selected_rows, axis=0)
        new_k = [int(it) for it in k_list]
        # 记录当前样本的长度
        l = len(new_k)
        # 原始序列（未掩码，便于缓存），按 seq_len 截断并填充
        original_seq = new_k[:self.seq_len]
        original_seq = pad_lists_to_length([original_seq], self.seq_len)[0]



        # else:
        k_masked, k_label= self.random_item(new_k)
        # k = k_masked
        # k_label = k_label
        k = k_masked
        k_label = k_label

        k = k[:self.seq_len]
        k_label = k_label[:self.seq_len]
        # k_label = [self.vocab.sos_index] + k_label
        mask = [1]*len(k)

        # 接下来对其进行填充操作
        seq_len = self.seq_len
        bert_input = k[:seq_len]
        bert_label = k_label[:seq_len]

        if len(mask) < seq_len:
            mask += [0] * (seq_len - len(mask))
        # 如果不足，就进行填充
        bert_input, bert_label= pad_lists_to_length(
            [bert_input, bert_label],
            seq_len
        )
        # 紧接着,需要将k改为已嵌入的方式
        bert_input = [x - 1 for x in bert_input]
        k_in = []

        # 创建一个全零数组，128维度
        zero_array = np.zeros(self.dim)

        # print(zero_array)

        for elem in bert_input:
            if elem == self.cls_index-1:
                k_in.append(cls_embedding)
            elif elem == self.mask_index -1:
                k_in.append(self.mask_embedding)
            elif elem == self.pad_index -1:
                k_in.append(zero_array)
            else:
                k_in.append(self.embedding_arr[elem])

        cls_label = [self.vocab_dict.get(k, self.vocab_size+1) if k != 0 else 0 for k in bert_label]

        cls_label = torch.tensor(cls_label, dtype=torch.long)
        k = torch.tensor(k_in, dtype=torch.float)
        k_label = torch.tensor(bert_label, dtype=torch.int)
        mask = torch.tensor(mask)
        original_seq = torch.tensor(original_seq, dtype=torch.long)
        return k, k_label, cls_label, mask, l, original_seq

    def random_item(self, k):
        tokens = k
        output_label = []




        for i, token in enumerate(tokens):

            if self.flag == "odd":
                if (i+1)%2!=0:
                    output_label.append(token)
                    tokens[i] = self.mask_index
                else:
                    output_label.append(0)
            else:
                if (i+1)%2 == 0:
                    output_label.append(token)
                    tokens[i] = self.mask_index

                else:
                    output_label.append(0)



        return tokens, output_label

