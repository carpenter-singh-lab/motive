import numpy as np
import pandas as pd

SEED = [2023, 7, 12]


def split_per_column_value(
    edges: pd.DataFrame, leave_out: str, subset_col: str = "subset"
):
    edges = edges.copy()
    quantile_col = "__quantile"
    edges[quantile_col] = pd.qcut(edges[leave_out], 10).cat.codes + 1
    subset = {}
    tile = np.repeat(["test", "valid", "train"], [2, 1, 7])
    rng = np.random.default_rng(SEED)
    for q, sub in edges.groupby(quantile_col):
        rng.shuffle(tile)
        counts = sub[leave_out].value_counts()
        n_repeat = len(counts) // 4 + 1
        tags = np.tile(tile, n_repeat)
        rng.shuffle(tags)
        subset.update(dict(zip(counts.index, tags)))
    edges[subset_col] = edges[leave_out].map(subset)
    edges.drop(columns=[quantile_col], inplace=True)
    index = edges.query(f'{subset_col}=="train"').index.values
    rng.shuffle(index)
    message_frac = 0.6
    num_message_edges = int(len(index) * message_frac)
    edges.loc[index[:num_message_edges], subset_col] = "message"
    return edges


def split_same_type(edges: pd.DataFrame, s_t_edges: pd.DataFrame, leave_out: str):
    edges["subset"] = "train"
    col1 = leave_out + "_a"
    col2 = leave_out + "_b"

    valid_nodes = s_t_edges[leave_out][s_t_edges["subset"] == "valid"].unique()
    test_nodes = s_t_edges[leave_out][s_t_edges["subset"] == "test"].unique()

    edges.loc[edges[col1].isin(valid_nodes), "subset"] = "valid"
    edges.loc[edges[col2].isin(valid_nodes), "subset"] = "valid"

    edges.loc[edges[col1].isin(test_nodes), "subset"] = "test"
    edges.loc[edges[col2].isin(test_nodes), "subset"] = "test"

    # assert that no test nodes appear in same type edges to avoid data leakage
    non_test = edges[edges["subset"] != "test"]
    assert np.count_nonzero(non_test[col1].isin(test_nodes)) == 0
    assert np.count_nonzero(non_test[col2].isin(test_nodes)) == 0

    return edges


def split_bipartite_edges(edges: pd.DataFrame, leave_out: str):
    if leave_out == "random":
        rng = np.random.default_rng(SEED)
        subset = rng.choice(
            size=len(edges),
            a=["train", "message", "valid", "test"],
            p=[0.7 * 0.4, 0.7 * 0.6, 0.1, 0.2],
        )
        edges["subset"] = subset
    else:
        edges = split_per_column_value(edges, leave_out)
    return edges


def split_s_expanded_edges(
    s_s_edges: pd.DataFrame,
    s_t_edges: pd.DataFrame,
    leave_out: str,
):
    """
    Only add in source/source edges.
    """
    # if random, only need to partition source_target edges into train/valid/test
    if leave_out == "random":
        rng = np.random.default_rng(SEED)
        subset = rng.choice(
            size=len(s_t_edges),
            a=["train", "message", "valid", "test"],
            p=[0.7 * 0.4, 0.7 * 0.6, 0.1, 0.2],
        )
        s_t_edges["subset"] = subset
        s_s_edges["subset"] = "train"

    # if source or target, need to remove the left out s_s or t_t edges
    else:
        s_t_edges = split_per_column_value(s_t_edges, leave_out)
        if leave_out == "source":
            s_s_edges = split_same_type(
                s_s_edges, s_t_edges[["source", "subset"]], leave_out
            )
        elif leave_out == "target":
            s_s_edges["subset"] = "train"

    return s_s_edges, s_t_edges


def split_t_expanded_edges(
    s_t_edges: pd.DataFrame,
    t_t_edges: pd.DataFrame,
    leave_out: str,
):
    """
    Only add in target/target edges.
    """
    # if random, only need to partition source_target edges into train/valid/test
    if leave_out == "random":
        rng = np.random.default_rng(SEED)
        subset = rng.choice(
            size=len(s_t_edges),
            a=["train", "message", "valid", "test"],
            p=[0.7 * 0.4, 0.7 * 0.6, 0.1, 0.2],
        )
        s_t_edges["subset"] = subset
        t_t_edges["subset"] = "train"

    # if source or target, need to remove the left out s_s or t_t edges
    else:
        s_t_edges = split_per_column_value(s_t_edges, leave_out)
        if leave_out == "source":
            t_t_edges["subset"] = "train"

        elif leave_out == "target":
            t_t_edges = split_same_type(
                t_t_edges, s_t_edges[["target", "subset"]], leave_out
            )

    return s_t_edges, t_t_edges


def split_edges(
    s_s_edges: pd.DataFrame,
    s_t_edges: pd.DataFrame,
    t_t_edges: pd.DataFrame,
    leave_out: str,
):
    """
    Add in both source/source and target/target edges.
    """
    # if random, only need to partition source_target edges into train/valid/test
    if leave_out == "random":
        rng = np.random.default_rng(SEED)
        subset = rng.choice(
            size=len(s_t_edges),
            a=["train", "message", "valid", "test"],
            p=[0.7 * 0.4, 0.7 * 0.6, 0.1, 0.2],
        )
        s_t_edges["subset"] = subset
        s_s_edges["subset"] = "train"
        t_t_edges["subset"] = "train"

    # if source or target, need to remove the left out s_s or t_t edges
    else:
        s_t_edges = split_per_column_value(s_t_edges, leave_out)
        if leave_out == "source":
            s_s_edges = split_same_type(
                s_s_edges, s_t_edges[["source", "subset"]], leave_out
            )
            t_t_edges["subset"] = "train"

        elif leave_out == "target":
            s_s_edges["subset"] = "train"
            t_t_edges = split_same_type(
                t_t_edges, s_t_edges[["target", "subset"]], leave_out
            )

    return s_s_edges, s_t_edges, t_t_edges
