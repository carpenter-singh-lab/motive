import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.autonotebook import tqdm
import pandas as pd
from utils.evaluate import Evaluator, get_best_th, save_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2024313


def run_update(model, optimizer, data):
    optimizer.zero_grad()

    data.to(DEVICE)
    logits = model(data)
    ground_truth = data["binds"].edge_label
    unique_mapping = (ground_truth * 2 + (logits > 0)).to(torch.long)
    counts = torch.bincount(unique_mapping, minlength=4).reshape(2, 2).to(DEVICE)
    loss = F.binary_cross_entropy_with_logits(logits, ground_truth)
    loss.backward()
    optimizer.step()
    loss_value = float(loss) * logits.numel()
    num_examples = logits.numel()
    return counts, loss_value, num_examples


def run_train_epoch(model, train_loader, optimizer, writer, epoch):
    total_loss = total_examples = 0
    cm_counts = torch.zeros((2, 2), dtype=torch.long).to(DEVICE)
    for sampled_data in tqdm(train_loader, leave=False, disable=True):
        counts, loss_value, num_examples = run_update(model, optimizer, sampled_data)
        cm_counts += counts
        total_loss += loss_value
        total_examples += num_examples
    loss = total_loss / total_examples
    writer.add_scalar("Loss/train", loss, epoch)
    writer.flush()


def run_eval_epoch(model, val_loader, writer, epoch):
    logits = []
    ground_truth = []
    with torch.no_grad():
        for sampled_data in val_loader:
            sampled_data.to(DEVICE)
            logits.append(model(sampled_data))
            ground_truth.append(sampled_data["binds"].edge_label)
        logits = torch.cat(logits, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, ground_truth)
        writer.add_scalar("Loss/valid", loss, epoch)
        ground_truth = ground_truth.to(torch.int32)
        e = Evaluator("configs/eval/validation_evaluation_params.json")
        valid_metrics = e.evaluate(logits, ground_truth)

    for metric, score in valid_metrics.items():
        writer.add_scalar("Valid_" + metric, score, epoch)

    return ground_truth, logits, valid_metrics


def run_test(model, test_loader, best_th, writer):
    logits = []
    ground_truth = []
    src_ids = []
    tgt_ids = []
    with torch.no_grad():
        for sampled_data in test_loader:
            sampled_data.to(DEVICE)
            logits.append(model(sampled_data))
            ground_truth.append(sampled_data["binds"].edge_label)

            # sampled batch source and target ids of each edge in test set
            test_srcs = sampled_data["binds"].edge_label_index[0]
            test_tgts = sampled_data["binds"].edge_label_index[1]

            # global source and target ids of each edge in test set
            src_ids.append(sampled_data["source"].node_id[test_srcs])
            tgt_ids.append(sampled_data["target"].node_id[test_tgts])

        # save logits, scores, bool predictions, gt, and indices of srcs and tgts
        logits = torch.cat(logits, dim=0)
        preds = logits > best_th
        ground_truth = torch.cat(ground_truth, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, ground_truth)
        scores = torch.sigmoid(logits).cpu().numpy()
        sources = torch.cat(src_ids, dim=0).cpu().numpy()
        targets = torch.cat(tgt_ids, dim=0).cpu().numpy()
        writer.add_scalar("Loss/test", loss)

        ground_truth = ground_truth.to(torch.int32)
        e = Evaluator("configs/eval/test_evaluation_params.json")
        test_metrics = e.evaluate(logits, ground_truth)

    # save all to results table
    results = pd.DataFrame(sources, columns=["source"])
    results["target"] = targets
    results["score"] = scores
    results["y_pred"] = preds.cpu().numpy()
    results["y_true"] = ground_truth.cpu().numpy()

    results.sort_values(by=["score"], ascending=False, inplace=True)
    results["percentile"] = results.score.rank(pct=True)
    results.set_index(["source", "target"], inplace=True)

    return results, test_metrics


def log_gradients_in_model(model, writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            writer.add_histogram(tag + "/grad", value.grad.cpu(), step)


def train_loop(
    model,
    locator,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    log_gradients=False,
):
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=locator.config["learning_rate"],
        weight_decay=locator.config["weight_decay"],
    )
    writer = SummaryWriter(
        log_dir=locator.summary_path, comment=locator.config["model_name"]
    )

    if num_epochs < 1:
        raise ValueError("Invalid number of epochs")
    ground_truth, logits = None, None
    best_metric = 0
    for epoch in tqdm(range(1, num_epochs + 1)):
        run_train_epoch(model, train_loader, optimizer, writer, epoch)
        curr_gt, curr_logits, val_metrics = run_eval_epoch(
            model, val_loader, writer, epoch
        )
        if val_metrics["F1"] > best_metric:
            best_metric = val_metrics["F1"]
            ground_truth, logits = curr_gt, curr_logits
            best_th = get_best_th(logits, ground_truth)
            state = dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                best_th=best_th,
            )
            torch.save(state, locator.model_path)
            save_metrics(val_metrics, locator.valid_metrics_path)

        if log_gradients:
            log_gradients_in_model(model, writer, epoch)

    print("Best validation metric: " + str(best_metric))
    results, test_scores = run_test(model, test_loader, best_th, writer)
    return results, test_scores, best_th
