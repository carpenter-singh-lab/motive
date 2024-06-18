import pandas as pd

from . import split


def bipartite(labels_path, leave_out, split_path):
    edges = pd.read_parquet(labels_path)
    edges = split.split_bipartite_edges(edges, leave_out)
    edges.to_parquet(split_path)


def s_expanded(ss_labels_path, st_labels_path, leave_out, split_ss_path, split_st_path):
    ss_edges = pd.read_parquet(ss_labels_path)
    st_edges = pd.read_parquet(st_labels_path)
    ss_edges, st_edges = split.split_s_expanded_edges(ss_edges, st_edges, leave_out)
    ss_edges.to_parquet(split_ss_path)
    st_edges.to_parquet(split_st_path)


def t_expanded(st_labels_path, tt_labels_path, leave_out, split_st_path, split_tt_path):
    st_edges = pd.read_parquet(st_labels_path)
    tt_edges = pd.read_parquet(tt_labels_path)
    st_edges, tt_edges = split.split_t_expanded_edges(st_edges, tt_edges, leave_out)
    st_edges.to_parquet(split_st_path)
    tt_edges.to_parquet(split_tt_path)


def st_expanded(
    ss_labels_path,
    st_labels_path,
    tt_labels_path,
    leave_out,
    split_ss_path,
    split_st_path,
    split_tt_path,
):
    ss_edges = pd.read_parquet(ss_labels_path)
    st_edges = pd.read_parquet(st_labels_path)
    tt_edges = pd.read_parquet(tt_labels_path)
    ss_edges, st_edges, tt_edges = split.split_edges(
        ss_edges, st_edges, tt_edges, leave_out
    )
    ss_edges.to_parquet(split_ss_path)
    st_edges.to_parquet(split_st_path)
    tt_edges.to_parquet(split_tt_path)
