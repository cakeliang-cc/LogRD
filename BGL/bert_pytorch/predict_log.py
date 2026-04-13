import pickle
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert_pytorch.dataset.log_predict_dataset import LogDatasetTest
from bert_pytorch.dataset.sample import fixed_window
import torch.nn.functional as F

def compute_anomaly(results, params, seq_threshold=0.5):
    is_logkey = params["is_logkey"]
    total_errors = 0
    for seq_res in results:
        # label pairs as anomaly when over half of masked tokens are undetected
        if (is_logkey and seq_res["undetected_tokens"] > seq_res["masked_tokens"] * seq_threshold) :
            total_errors += 1
    return total_errors


def find_best_threshold(test_normal_results, test_abnormal_results, params, th_range, seq_range):
    best_result = [0] * 9
    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_results, params, seq_th)
        TP = compute_anomaly(test_abnormal_results, params, seq_th)

        if TP == 0:
            continue

        TN = len(test_normal_results) - FP
        FN = len(test_abnormal_results) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        if F1 > best_result[-1]:
            best_result = [0, seq_th, FP, TP, TN, FN, P, R, F1]
    return best_result


class Predictor():
    def __init__(self, options):
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.device = options["device"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.seq_len = options["seq_len"]
        self.corpus_lines = options["corpus_lines"]
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.num_candidates = options["num_candidates"]
        self.output_dir = options["output_dir"]
        self.model_dir = options["model_dir"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale_path = options["scale_path"]


        self.lower_bound = self.gaussian_mean - 3 * self.gaussian_std
        self.upper_bound = self.gaussian_mean + 3 * self.gaussian_std

        self.test_ratio = options["test_ratio"]
        self.mask_ratio = options["mask_ratio"]
        # 对于异常检测而言，可以利用长度直接判断是正常的还是异常的
        self.min_len= 0
        self.min_len_a = options["min_len"]
        self.min_len_b = 5
        self.normal_test_csv_path = options["normal_test"]
        self.anomaly_test_csv_path = options["anomaly_test"]
        self.embedding_path = options["embedding_path"]
        self.embedding_arr = np.load(self.embedding_path)
        self.threshold1 = options["threshold1"]
        self.vocab = pd.read_csv(options["train_vocab"])
        self.vocab_size = options["vocab_size"]
        self.no_replace_path = options["no_replace_path"]
        self.ab_replace_path = options["ab_replace_path"]
        self.similarity = options["similarity"]


    def generate_test(self, test_csv_path, replace_path, window_size, adaptive_window, seq_len, scale, min_len):
        """
        :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
        """
        content_id_list = self.vocab['contentId'].tolist()
        replace_df = pd.read_csv(replace_path)
        replace_df['replaceId'] = replace_df.apply(
            lambda row: row['train_key'] if row['similarity'] > self.similarity else row['test_key'],
            axis=1
        )
        # 将 replace_df 转换为字典，键是 'replaceId' 列，值是 'test_ab_result_dict_key' 列
        replace_dict = replace_df.set_index('test_key')['replaceId'].astype(int).to_dict()
        replace_dict = {str(k): str(v) for k, v in replace_dict.items()}
        log_seqs = []
        tim_seqs = []
        data_csv = pd.read_csv(test_csv_path)
        data_iter = data_csv.iloc[:, 1].tolist()
        for line in tqdm(data_iter):
            log_seq, tim_seq = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
            if len(log_seq) == 0:
                continue
            log_seqs += log_seq
            tim_seqs += tim_seq
        log_seqs = np.array(log_seqs, dtype=object)
        tim_seqs = np.array(tim_seqs, dtype=object)

        test_len = list(map(len, log_seqs))
        test_sort_index = np.argsort(-1 * np.array(test_len))

        log_seqs = log_seqs[test_sort_index]
        tim_seqs = tim_seqs[test_sort_index]

        for i in range(len(log_seqs)):
            log_seqs[i] = [replace_dict.get(x, x) for x in log_seqs[i]]
        print(f"{test_csv_path} size: {len(log_seqs)}")
        return log_seqs, tim_seqs

    def detect_logkey_anomaly(self, cls_output, cls_label):
        num_undetected_tokens = 0
        for i, token in enumerate(cls_label):
            if token not in torch.argsort(-cls_output[i])[:self.num_candidates]:
                num_undetected_tokens += 1

        return num_undetected_tokens


    def helper(self, model, test_csv_path ,  replace_path, scale=None):
        total_results = []

        output_cls = []
        output_cls2 = []

        logkey_test, time_test = self.generate_test(test_csv_path, replace_path, self.window_size,self.adaptive_window,self.seq_len,
                                                    scale,self.min_len)
        # use 1/10 test data
        if self.test_ratio != 1:
            num_test = len(logkey_test)
            rand_index = torch.randperm(num_test)
            rand_index = rand_index[:int(num_test * self.test_ratio)] if isinstance(self.test_ratio,
                                                                                    float) else rand_index[
                                                                                                :self.test_ratio]
            logkey_test, time_test = logkey_test[rand_index], time_test[rand_index]

        # 是不是需要在数据集的构建中添加当前的序列的长度
        seq_dataset = LogDatasetTest(self.vocab, logkey_test, time_test, self.embedding_arr, seq_len=self.seq_len,
                                   corpus_lines=self.corpus_lines, on_memory=self.on_memory, mask_ratio=self.mask_ratio)
        seq_dataset_even = LogDatasetTest(self.vocab, logkey_test, time_test, self.embedding_arr, seq_len=self.seq_len,
                                   corpus_lines=self.corpus_lines, on_memory=self.on_memory, mask_ratio=self.mask_ratio,flag="even")


        # 不采用多线程，保证顺序是一样的
        data_loader = DataLoader(seq_dataset, batch_size=self.batch_size, num_workers=0)
        data_loader_even = DataLoader(seq_dataset_even, batch_size=self.batch_size, num_workers=0)

        for idx, (data, data2) in enumerate(zip(data_loader, data_loader_even)):
            # 获得的是批次的每一个的长度
            l_juzhen = data[4]

            # 0是输入数据嵌入
            data[0] = data[0].to(self.device)
            # 1是回归的标签
            data[1] = data[1].to(self.device)
            # 2是分类的标签
            data[2] = data[2].to(self.device)
            # 3是掩码矩阵
            data[3] = data[3].to(self.device)
            result = model(data[0], data[3])

            cls_output = result["cls_output"]

            data2[0] = data2[0].to(self.device)
            data2[1] = data2[1].to(self.device)
            data2[2] = data2[2].to(self.device)
            data2[3] = data2[3].to(self.device)
            result2 = model(data2[0], data2[3])

            cls_output2 = result2["cls_output"]



            for i in range(len(data[1])):

                l = l_juzhen[i]
                # 针对批次中的每一个日志序列，构造一个字典，记录字典的结果
                seq_results = {"num_error": 0,
                               "undetected_tokens": 0,
                               "masked_tokens": 0,
                               "deepSVDD_label": 0
                               }
                mask_index = data[1][i] > 0
                num_masked = torch.sum(mask_index).tolist()
                seq_results["masked_tokens"] = num_masked

                mask_index2 = data2[1][i] > 0
                num_masked2 = torch.sum(mask_index2).tolist()
                seq_results["masked_tokens"]+=num_masked2

                if l<self.min_len_a:
                    seq_results["undetected_tokens"] = seq_results["masked_tokens"]
                    continue



                if self.is_logkey:
                    # 如果是日志键的话，根据阈值判断正常还是异常
                    num_undetected = self.detect_logkey_anomaly(cls_output[i][mask_index], data[2][i][mask_index])
                    num_undetected2 = self.detect_logkey_anomaly(cls_output2[i][mask_index2], data2[2][i][mask_index2])
                    seq_results["undetected_tokens"] = num_undetected+num_undetected2




                if idx < 10 or idx % 1000 == 0:
                    print(
                        "{}, #time anomaly: {} # of undetected_tokens: {}, # of masked_tokens: {} , "
                        "# deepSVDD_label: {} \n".format(
                            test_csv_path,
                            seq_results["num_error"],
                            seq_results["undetected_tokens"],
                            seq_results["masked_tokens"],
                            # seq_results["total_logkey"],
                            seq_results['deepSVDD_label']
                        )
                    )
                total_results.append(seq_results)
        return total_results, output_cls




    def predict(self):


        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))
        start_time = time.time()
        scale = None
        error_dict = None
        if self.is_time:
            with open(self.scale_path, "rb") as f:
                scale = pickle.load(f)

            with open(self.model_dir + "error_dict.pkl", 'rb') as f:
                error_dict = pickle.load(f)



        print("test normal predicting")
        test_normal_results, test_normal_errors = self.helper(model,self.normal_test_csv_path, self.no_replace_path, scale)
        print("test abnormal predicting")
        test_abnormal_results, test_abnormal_errors = self.helper(model,self.anomaly_test_csv_path, self.ab_replace_path, scale)

        print("Saving test normal results")
        with open(self.model_dir + "test_normal_results", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_results", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        print("Saving test normal errors")
        with open(self.model_dir + "test_normal_errors.pkl", "wb") as f:
            pickle.dump(test_normal_errors, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_errors.pkl", "wb") as f:
            pickle.dump(test_abnormal_errors, f)

        params = {"is_logkey": self.is_logkey, "is_time": self.is_time}



        best_th, best_seq_th, FP, TP, TN, FN, P, R, F1 = find_best_threshold(test_normal_results,
                                                                             test_abnormal_results,
                                                                             params=params,
                                                                             th_range=np.arange(10),
                                                                             seq_range=np.arange(0, 1, 0.05))

        print("best threshold: {}, best threshold ratio: {}".format(best_th, best_seq_th))
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))


