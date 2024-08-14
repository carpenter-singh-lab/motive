import argparse
import os.path

from model import create_model
from motive import get_loaders
from train import DEVICE, run_test, train_loop
from utils.evaluate import save_metrics
from utils.utils import PathLocator


def workflow(locator, num_epochs, tgt_type, graph_type, eval_test=False):
    leave_out = locator.config["data_split"]
    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type)
    train_data = train_loader.loader.data
    model = create_model(locator, train_data).to(DEVICE)
    best_th = train_loop(model, locator, train_loader, val_loader, num_epochs)
    if eval_test:
        results, test_scores = run_test(model, test_loader, best_th)
        save_metrics(test_scores, locator.test_metrics_path)
        results.to_parquet(locator.test_results_path)
        print(test_scores)


def main():
    """Parse input params"""
    parser = argparse.ArgumentParser(
        description=("Train GNN with this config file"),
    )

    parser.add_argument("config_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000)

    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")
    args = parser.parse_args()

    locator = PathLocator(args.config_path, args.output_path)
    if os.path.isfile(locator.test_results_path):
        print(f"{locator.test_results_path} exists. Skipping...")
        return
    workflow(
        locator, args.num_epochs, args.target_type, args.graph_type, eval_test=True
    )


if __name__ == "__main__":
    main()
