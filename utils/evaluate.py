import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torcheval.metrics.functional.ranking import retrieval_precision
from torcheval.metrics.functional import binary_f1_score


def get_best_th(logits, y_true):
    if torch.is_tensor(logits) or torch.is_tensor(y_true):
        logit_values = torch.unique(logits)
        ths = torch.rand(50, device=logits.device.type)
        ths = ths * (logit_values[-1] - logit_values[0]) + logit_values[0]
        f1s = []
        for th in ths:
            f1s.append(binary_f1_score(logits, y_true, threshold=th))
        ind = torch.argsort(torch.FloatTensor(f1s))
        best_th = ths[ind][-1]

    else:
        logit_values = np.unique(logits)
        values = {
            th: f1_score(y_true, logits > th)
            for th in np.random.uniform(logit_values[0], logit_values[-1], size=50)
        }
        best_th = pd.Series(values).sort_values().index[-1]

    return best_th


class Evaluator:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf8") as freader:
            self.config = json.load(freader)
        self.scores = {}

    def evaluate(self, logits, y_true):
        for metric in self.config["metrics"]:
            if metric == "ACC":
                score = self._eval_ACC(logits, y_true)

            elif metric == "AUC":
                score = self._eval_AUC(logits, y_true)

            elif metric == "Hits@K":
                metric = "Hits@{}".format(self.config["Hits_K"])
                score = self._eval_hits_at_K(logits, y_true)

            elif metric == "Precision@K":
                metric = "Precision@{}".format(self.config["Precision_K"])
                score = self._eval_precision_at_K(logits, y_true)

            elif metric == "F1":
                score = self._eval_F1(logits, y_true)

            elif metric == "MRR":
                score = self._eval_MRR(logits, y_true)

            self.scores[metric] = score

        return self.scores

    def _eval_ACC(self, logits, y_true):
        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            logits = logits.cpu().numpy()
            y_true = y_true.cpu().numpy()

        best_th = get_best_th(logits, y_true)
        y_pred = logits > best_th
        return accuracy_score(y_true, y_pred)

    def _eval_AUC(self, logits, y_true):
        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            logits = logits.cpu().numpy()
            y_true = y_true.cpu().numpy()

        y_pred = 1 / (1 + np.exp(-logits))
        return roc_auc_score(y_true, y_pred)

    def _eval_hits_at_K(self, logits, y_true):
        k = self.config["Hits_K"]

        logits_pos = logits[y_true == 1]
        logits_neg = logits[y_true == 0]

        if len(logits_neg) < k:
            return 1.0

        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            kth_score_in_negative_edges = torch.topk(logits_neg, k)[0][-1]
            hitsK = float(
                torch.sum(logits_pos > kth_score_in_negative_edges).cpu()
            ) / len(logits_pos)

        else:
            kth_score_in_negative_edges = np.sort(logits_neg)[-k]
            hitsK = float(np.sum(logits_pos >= kth_score_in_negative_edges)) / len(
                logits_pos
            )

        return hitsK

    def _eval_precision_at_K(self, logits, y_true):
        k = self.config["Precision_K"]

        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            precision = retrieval_precision(logits, y_true, k).item()

        else:
            rank_order = np.argsort(logits)
            ranked_logits = y_true[rank_order][-k:]
            precision = np.sum(ranked_logits) / k
        return precision

    def _eval_F1(self, logits, y_true):
        best_th = get_best_th(logits, y_true)
        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            best_th = torch.sigmoid(best_th).item()
            y_pred = torch.sigmoid(logits)
            f1 = binary_f1_score(y_pred, y_true, threshold=best_th).item()
        else:
            best_th = 1 / (1 + np.exp(-best_th))[0]
            scores = 1 / (1 + np.exp(-logits))
            y_pred = scores > best_th
            f1 = f1_score(y_true, y_pred, zero_division=0)

        return f1

    def _eval_MRR(self, logits, y_true):
        logits_pos = logits[y_true == 1]
        logits_neg = logits[y_true == 0]

        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            y_pred_pos = logits_pos.view(-1, 1)
            optimistic_rank = (logits_neg > y_pred_pos).sum(dim=1)
            pessimistic_rank = (logits_neg >= y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            mrr_list = 1.0 / ranking_list.to(torch.float)
            mrr = torch.mean(mrr_list).item()

        else:
            logits_pos = logits_pos.reshape(-1, 1)
            optimistic_rank = (logits_neg > logits_pos).sum(axis=1)
            pessimistic_rank = (logits_neg >= logits_pos).sum(axis=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

            mrr_list = 1.0 / ranking_list.astype(np.float32)
            mrr = np.mean(mrr_list)

        return mrr


def save_metrics(scores: dict, output_path: str):
    pd.DataFrame.from_dict(scores, "index", dtype=np.float32).to_csv(
        output_path, index=True, header=None
    )
