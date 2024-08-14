import json
import warnings
from bisect import bisect_left

import numpy as np
import pandas as pd
import torch
from copairs import compute
from copairs.map.average_precision import build_rank_lists, p_values
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn import functional as F
from torcheval.metrics.functional import binary_f1_score
from torcheval.metrics.functional.ranking import retrieval_precision


def compute_map(df):
    """
    input: dframe with "source", "target", "y_true", "score" columns
    output: ap_scores
    """
    src_ix = df["source"].unique()
    tgt_ix = df["target"].unique()
    num_src, num_tgt = len(src_ix), len(tgt_ix)
    src_mapper = dict(zip(src_ix, range(num_src)))
    tgt_mapper = dict(zip(tgt_ix, range(num_src, num_src + num_tgt)))
    df["src_copairs_id"] = df["source"].map(src_mapper)
    df["tgt_copairs_id"] = df["target"].map(tgt_mapper)
    pos_pairs = df.query("y_true==1")[["src_copairs_id", "tgt_copairs_id"]].values
    neg_pairs = df.query("y_true==0")[["src_copairs_id", "tgt_copairs_id"]].values
    pos_sims = df.query("y_true==1")["score"].values
    neg_sims = df.query("y_true==0")["score"].values
    paired_ix, rel_k_list, counts = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims
    )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value encountered in divide')
        ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
    ap_scores = pd.DataFrame(
        {
            "node_id": np.concatenate([src_ix, tgt_ix]),
            "node_type": ["source"] * num_src + ["target"] * num_tgt,
            "average_precision": ap_scores,
            "n_pos_pairs": null_confs[:, 0],
            "n_total_pairs": null_confs[:, 1],
        }
    )
    ap_scores["p_value"] = p_values(ap_scores, 10000, 0)
    ap_scores["below_p_value"] = ap_scores["p_value"] < 0.05
    return ap_scores


def get_best_th(logits, y_true):
    return 0
    # TODO: Check binary f1 is unimodal so that binary search fits
    ths = torch.unique(logits).cpu().numpy()
    best_f1 = 1
    ix = bisect_left(
        ths, -best_f1, key=lambda th: -binary_f1_score(logits, y_true, threshold=th)
    )
    return ths[ix]


class Evaluator:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf8") as freader:
            self.config = json.load(freader)
        self.scores = {}

    def evaluate(self, logits, y_true, th, edges):
        for metric in self.config["metrics"]:
            if metric == "ACC":
                score = self._eval_ACC(logits, y_true, th)

            elif metric == "AUC":
                score = self._eval_AUC(logits, y_true)

            elif metric == "Hits@K":
                metric = "Hits@{}".format(self.config["Hits_K"])
                score = self._eval_hits_at_K(logits, y_true)

            elif metric == "Precision@K":
                metric = "Precision@{}".format(self.config["Precision_K"])
                score = self._eval_precision_at_K(logits, y_true)

            elif metric == "F1":
                score = self._eval_F1(logits, y_true, th)

            elif metric == "MRR":
                score = self._eval_MRR(logits, y_true)

            elif metric == "BCELoss":
                score = self._eval_loss(logits, y_true)
            elif metric == "mAP":
                score = self._eval_mAP(logits, y_true, edges)
            if type(score) is dict:
                self.scores.update(score)
            else:
                self.scores[metric] = float(score)

        return self.scores

    def _eval_ACC(self, logits, y_true, th):
        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            logits = logits.cpu().numpy()
            y_true = y_true.cpu().numpy()

        y_pred = logits > th
        return accuracy_score(y_true, y_pred)

    def _eval_AUC(self, logits, y_true):
        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            logits = logits.cpu().numpy()
            y_true = y_true.cpu().numpy()

        y_pred = 1 / (1 + np.exp(-np.clip(logits, 1e-5, 1e5)))
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

    def _eval_F1(self, logits, y_true, th):
        if torch.is_tensor(logits) or torch.is_tensor(y_true):
            f1 = binary_f1_score(logits, y_true, threshold=th).item()
        else:
            f1 = f1_score(y_true, logits > th, zero_division=0)
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

    @torch.inference_mode
    def _eval_loss(self, logits, y_true):
        return F.binary_cross_entropy_with_logits(logits, y_true.to(torch.float32))

    def _eval_mAP(self, logits, y_true, edges):
        source, target = edges.cpu().numpy()
        dframe = pd.DataFrame(
            {
                "source": source,
                "target": target,
                "y_true": y_true.cpu().numpy(),
                "score": logits.cpu().numpy(),
            }
        )
        split = "source"
        ap_scores = compute_map(dframe).query(f"node_type=='{split}'")
        lt_p = ap_scores.below_p_value.sum()
        lt_p_ratio = lt_p / len(ap_scores)
        mean_average_precision = ap_scores["average_precision"].mean()
        scores = {
            f"{split}_lt_p": lt_p,
            f"{split}_lt_p_ratio": lt_p_ratio,
            f"{split}_mAP": mean_average_precision,
        }
        return scores


def save_metrics(scores: dict, output_path: str):
    pd.DataFrame.from_dict(scores, "index", dtype=np.float32).to_csv(
        output_path, index=True, header=None
    )
