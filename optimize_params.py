import argparse
import json
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter

from plot.plot_exploration import contour
from run_training import workflow as train_workflow
from train import SEED
from utils.utils import PathLocator


def generate_parameters(num_opts: int):
    config_search = []
    rng = np.random.default_rng(SEED)
    for i in range(num_opts):
        hidden_channels = rng.choice([64, 128, 256])
        learning_rate = 10.0 ** rng.uniform(-6, -2)
        weight_decay = 10.0 ** rng.uniform(-5, 1)

        config_search.append((hidden_channels, learning_rate, weight_decay))

    config_search_df = pd.DataFrame(
        config_search, columns=["hidden_channels", "learning_rate", "weight_decay"]
    )

    config_search_df.to_csv("configs/optimize/optimize_configs.csv", index=False)
    return config_search_df


def main():
    """Parse input params"""
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("--i", dest="initialization", type=str)
    parser.add_argument("num_search", type=int)
    parser.add_argument("data_split", type=str)
    parser.add_argument(
        "--generate_new_params", dest="generate_new", action="store_true", default=False
    )
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000)
    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")

    args = parser.parse_args()

    # generate new random search parameters
    if args.generate_new:
        configs = generate_parameters(args.num_search)

    # or load previously generated parameters
    else:
        configs = pd.read_csv("configs/optimize/optimize_configs.csv")

    # save results in df
    results = []

    # decide how many parameter combinations to search
    for curr_config in configs.iloc[: args.num_search].itertuples():
        curr_config = curr_config._asdict()
        del curr_config["Index"]
        curr_config["model_name"] = args.model
        curr_config["data_split"] = args.data_split

        if args.model in ("gnn", "gat", "gin"):
            curr_config["initialization"] = args.initialization

        with TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/config.json"
            with open(filename, "w", encoding="utf8") as fout:
                json.dump(curr_config, fout)
            locator = PathLocator(filename, args.output_path)
            curr_config["config_id"] = locator.hashid

        if locator.model_path.is_file():
            num_epochs = 0
        else:
            num_epochs = args.num_epochs
        train_workflow(
            locator,
            num_epochs,
            args.target_type,
            args.graph_type,
        )
        metrics = pd.read_csv(locator.valid_metrics_path, header=None)
        curr_config.update(metrics.set_index(0)[1])
        results.append(curr_config)

    criteria = "Hits@500"
    results = pd.json_normalize(results).sort_values(by=criteria, ascending=False)
    results.to_csv(f"{args.output_path}/param_search.csv", index=False)
    writer = SummaryWriter(log_dir=args.output_path)
    writer.add_image(
        f"{criteria} exploration",
        contour(results, criteria),
        dataformats="HWC",
    )
    writer.flush()
    print(results)


if __name__ == "__main__":
    main()
