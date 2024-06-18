import numpy as np
import pandas as pd


def select_features(dframe: pd.DataFrame, column_id: str) -> pd.DataFrame:
    """
    Helper function to strip excess metadata from profile dataframes.
    """
    featcols = [c for c in dframe.columns if not c.startswith("Meta")]
    dframe = dframe.set_index(column_id).sort_index()
    dframe = dframe[featcols]
    return dframe


def map_broad_to_inchi(dframe: pd.DataFrame, source_map_path: str):
    """
    Helper function to add InChI Keys to source dataframe.
    """
    compound_metadata = pd.read_csv(source_map_path, compression="gzip")
    compound_metadata.drop_duplicates("Metadata_InChIKey", inplace=True)
    mapper = pd.Series(
        compound_metadata["Metadata_InChIKey"].str[:14].values,
        compound_metadata["Metadata_JCP2022"].values,
    )

    # map the broad IDs to their corresponding InChIKeys
    inchi_col = dframe["Metadata_JCP2022"].map(mapper)

    # add the InChIKey column to the corrected data table
    dframe.insert(4, "Metadata_InChIKey", inchi_col)
    dframe = dframe[dframe["Metadata_InChIKey"].notna()]

    return dframe


def map_broad_to_symbol(dframe: pd.DataFrame, target_map_path: str):
    """
    Helper function to add Gene Symbols to target dataframe.
    """

    gene_metadata = pd.read_csv(target_map_path, compression="gzip")
    gene_metadata.drop_duplicates("Metadata_Symbol", inplace=True)
    mapper = pd.Series(
        gene_metadata["Metadata_Symbol"].str[:14].values,
        gene_metadata["Metadata_JCP2022"].values,
    )

    # map the broad IDs to their corresponding InChIKeys
    symbol_col = dframe["Metadata_JCP2022"].map(mapper)

    # add the InChIKey column to the corrected data table
    dframe.insert(4, "Metadata_Symbol", symbol_col)
    dframe = dframe[dframe["Metadata_Symbol"].notna()]

    return dframe


def reduce_edges(data: pd.DataFrame, max_degree: int):
    """
    Prune off edges so that each node fewer than the max_degree number
    of neighbors (approximately).
    """
    max_degree = int(max_degree)
    col1 = data.columns[0]
    col2 = data.columns[1]

    all_data = np.hstack((data[col1], data[col2]))
    unique_nodes, degrees = np.unique(all_data, return_counts=True)

    while np.max(degrees) > max_degree:
        new_nodes = unique_nodes[degrees < np.quantile(degrees, 0.99)]

        filtered_data = data[data[col1].isin(new_nodes)]
        filtered_data = filtered_data[filtered_data[col2].isin(new_nodes)]

        all_data = np.hstack((filtered_data[col1], filtered_data[col2]))
        unique_nodes, degrees = np.unique(all_data, return_counts=True)

    return filtered_data.reset_index(drop=True)


def load_jump_compounds(
    source_path: str,
    source_jcpid_path: str,
    all_source_path: str,
):
    """
    Get profiles from Jump compound data. Compute a median profile per compound.
    """
    sources = pd.read_parquet(source_path)
    sources = map_broad_to_inchi(sources, source_jcpid_path)
    metacols = [c for c in sources.columns if c.startswith("Meta")]
    featcols = [c for c in sources.columns if not c.startswith("Meta")]
    agg_funcs = {c: "first" for c in metacols}
    agg_funcs.update({c: "median" for c in featcols})
    profiles = sources.groupby("Metadata_JCP2022", observed=True).agg(agg_funcs)
    profiles.drop_duplicates("Metadata_InChIKey", inplace=True)
    profiles = select_features(profiles, "Metadata_InChIKey")
    profiles.to_parquet(all_source_path)


def load_jump_targets(target_path: str, meta_path: str, all_target_path: str):
    """
    Get profiles from Jump target data. Compute a median profile per reagent.
    When there are multiple reagents per gene, pick the first occurence.
    """
    feats = pd.read_parquet(target_path)
    feats = map_broad_to_symbol(feats, meta_path)

    metacols = [c for c in feats.columns if c.startswith("Meta")]
    featcols = [c for c in feats.columns if not c.startswith("Meta")]
    agg_funcs = {c: "first" for c in metacols}
    agg_funcs.update({c: "median" for c in featcols})
    profiles = feats.groupby("Metadata_Symbol", observed=True).agg(agg_funcs)
    profiles = select_features(profiles, "Metadata_Symbol")

    profiles.to_parquet(all_target_path)


def load_jump_annotations(annotation_path: str, all_labels_path: str):
    annotations = pd.read_parquet(annotation_path)
    annotations.dropna(subset=["inchikey", "target"], how="any", inplace=True)
    annotations["inchikey"] = annotations["inchikey"].str[:14]
    annotations.drop_duplicates(subset=["inchikey", "target"], inplace=True)
    annotations.rename(
        columns={"inchikey": "Metadata_InChIKey", "target": "Metadata_Symbol"},
        inplace=True,
    )
    annotations.reset_index(drop=True, inplace=True)
    annotations.to_parquet(all_labels_path)


def load_reduced_jump_annotations(
    annotation_path: str, all_s_t_labels_path: str, max_degree=150
):
    annotations = pd.read_parquet(annotation_path)
    annotations.dropna(subset=["inchikey", "target"], how="any", inplace=True)
    annotations["inchikey"] = annotations["inchikey"].str[:14]
    annotations.drop_duplicates(subset=["inchikey", "target"], inplace=True)
    annotations = reduce_edges(annotations, max_degree)
    annotations.rename(
        columns={"inchikey": "Metadata_InChIKey", "target": "Metadata_Symbol"},
        inplace=True,
    )
    annotations.reset_index(drop=True, inplace=True)
    annotations.to_parquet(all_s_t_labels_path)


def load_gene_gene_annotations(
    gene_gene_annotation_path: str,
    jump_genes_path: str,
    all_t_t_labels_path: str,
    max_degree=150,
):
    """
    Remove duplicates of all types from gene/gene annotations
    """
    dframe = pd.read_parquet(gene_gene_annotation_path)
    dframe.dropna(subset=["target_a", "target_b"], how="any", inplace=True)
    jump_targets = pd.read_parquet(jump_genes_path).index

    # (1) remove self loops
    dframe = dframe[dframe["target_a"] != dframe["target_b"]].reset_index(drop=True)

    # (2) remove exact duplicate pairs
    dframe = dframe.drop_duplicates(subset=["target_a", "target_b"], ignore_index=True)

    # (3) remove targest that are not in jump targets
    dframe = dframe[dframe["target_a"].isin(jump_targets)].reset_index(drop=True)
    dframe = dframe[dframe["target_b"].isin(jump_targets)].reset_index(drop=True)

    # (4) remove out-of-order duplicate pairs
    dupe_idxs = (
        dframe[["target_a", "target_b"]]
        .apply(lambda x: "-".join(sorted(x)), axis=1)
        .duplicated()
    )
    dframe = dframe.loc[~dupe_idxs].reset_index(drop=True)

    # (5) reduce nodes with degree > max_degree
    dframe = reduce_edges(dframe, max_degree)
    dframe.to_parquet(all_t_t_labels_path)


def load_compound_compound_annotations(
    compound_compound_annotation_path: str,
    jump_cmpds_path: str,
    all_s_s_labels_path: str,
    max_degree=150,
):
    """
    Remove duplicates of all types from gene/gene annotations
    """
    dframe = pd.read_parquet(compound_compound_annotation_path)
    dframe.dropna(subset=["inchikey_a", "inchikey_b"], how="any", inplace=True)
    dframe["inchikey_a"] = dframe["inchikey_a"].str[:14]
    dframe["inchikey_b"] = dframe["inchikey_b"].str[:14]

    jump_sources = pd.read_parquet(jump_cmpds_path).index

    dframe = dframe[["inchikey_a", "inchikey_b", "rel_type", "source_id", "database"]]
    dframe.rename(
        columns={"inchikey_a": "source_a", "inchikey_b": "source_b"}, inplace=True
    )

    dframe = dframe[dframe["source_a"] != dframe["source_b"]].reset_index(drop=True)
    dframe = dframe.drop_duplicates(subset=["source_a", "source_b"], ignore_index=True)
    dframe = dframe[dframe["source_a"].isin(jump_sources)].reset_index(drop=True)
    dframe = dframe[dframe["source_b"].isin(jump_sources)].reset_index(drop=True)
    dupe_idxs = (
        dframe[["source_a", "source_b"]]
        .apply(lambda x: "-".join(sorted(x)), axis=1)
        .duplicated()
    )
    dframe = dframe.loc[~dupe_idxs].reset_index(drop=True)
    dframe = reduce_edges(dframe, max_degree)
    dframe.to_parquet(all_s_s_labels_path)


def merge_bipartite(
    all_source_path: str,
    all_target_path: str,
    all_s_t_labels_path: str,
    source_path: str,
    target_path: str,
    labels_path: str,
    source_map_path: str,
    target_map_path: str,
):
    # merge the annotations with the jump sources and targets
    sources = pd.read_parquet(all_source_path)
    targets = pd.read_parquet(all_target_path)
    annotations = pd.read_parquet(all_s_t_labels_path)

    source_ids = pd.Series(sources.index)
    target_ids = pd.Series(targets.index)
    annotations = annotations.merge(source_ids, on="Metadata_InChIKey", copy=False)
    annotations = annotations.merge(target_ids, on="Metadata_Symbol", copy=False)

    # gives a list of targets for each source
    src_to_tgt = annotations.groupby("Metadata_InChIKey")["Metadata_Symbol"].apply(
        "|".join
    )

    src_to_tgt = src_to_tgt.str.split("|").explode().reset_index()
    src_to_tgt = src_to_tgt.drop_duplicates()

    # filter sources and targets by ones with annotations
    sources_with_links = src_to_tgt["Metadata_InChIKey"].unique()
    targets_with_links = src_to_tgt["Metadata_Symbol"].unique()
    sources = sources[sources.index.isin(sources_with_links)]
    targets = targets[targets.index.isin(targets_with_links)]

    # map from sources and targets to their ids
    source_map = pd.Series(range(len(sources)), index=sources.index)
    target_map = pd.Series(range(len(targets)), index=targets.index)

    pd.DataFrame(source_map).to_parquet(source_map_path)
    pd.DataFrame(target_map).to_parquet(target_map_path)

    edges = pd.DataFrame(
        {
            "source": src_to_tgt["Metadata_InChIKey"].map(source_map).values,
            "target": src_to_tgt["Metadata_Symbol"].map(target_map).values,
        }
    )

    sources.to_parquet(source_path)
    targets.to_parquet(target_path)
    edges.to_parquet(labels_path)


def merge_s_expanded(
    all_source_path: str,
    all_target_path: str,
    all_s_s_labels_path: str,
    all_s_t_labels_path: str,
    source_path: str,
    target_path: str,
    s_s_labels_path: str,
    s_t_labels_path: str,
    source_map_path: str,
    target_map_path: str,
):
    # merge the annotations with the jump sources and targets
    sources = pd.read_parquet(all_source_path)
    targets = pd.read_parquet(all_target_path)
    s_s_labels = pd.read_parquet(all_s_s_labels_path)
    s_t_labels = pd.read_parquet(all_s_t_labels_path)

    source_ids = pd.Series(sources.index)
    target_ids = pd.Series(targets.index)
    s_t_labels = s_t_labels.merge(source_ids, on="Metadata_InChIKey", copy=False)
    s_t_labels = s_t_labels.merge(target_ids, on="Metadata_Symbol", copy=False)

    # gives a list of targets for each source
    src_to_tgt = s_t_labels.groupby("Metadata_InChIKey")["Metadata_Symbol"].apply(
        "|".join
    )

    src_to_tgt = src_to_tgt.str.split("|").explode().reset_index()
    src_to_tgt = src_to_tgt.drop_duplicates()

    # filter sources and targets by ones with annotations (of any kind)
    # for bipartite, we will have 2683 sources and 3780 targets at this point
    sources_with_links = src_to_tgt["Metadata_InChIKey"].unique()
    targets_with_links = src_to_tgt["Metadata_Symbol"].unique()

    more_sources = (
        pd.concat([s_s_labels["source_a"], s_s_labels["source_b"]], axis=0)
        .unique()
        .astype(np.ndarray)
    )
    sources_with_links = np.unique(np.hstack((sources_with_links, more_sources)))

    sources = sources[sources.index.isin(sources_with_links)]
    targets = targets[targets.index.isin(targets_with_links)]

    # map from sources and targets to their ids
    source_map = pd.Series(range(len(sources)), index=sources.index)
    target_map = pd.Series(range(len(targets)), index=targets.index)

    pd.DataFrame(source_map).to_parquet(source_map_path)
    pd.DataFrame(target_map).to_parquet(target_map_path)

    s_t_edges = pd.DataFrame(
        {
            "source": src_to_tgt["Metadata_InChIKey"].map(source_map).values,
            "target": src_to_tgt["Metadata_Symbol"].map(target_map).values,
        }
    )
    s_s_edges = pd.DataFrame(
        {
            "source_a": s_s_labels["source_a"].map(source_map).values,
            "source_b": s_s_labels["source_b"].map(source_map).values,
        }
    )

    sources.to_parquet(source_path)
    targets.to_parquet(target_path)
    s_s_edges.to_parquet(s_s_labels_path)
    s_t_edges.to_parquet(s_t_labels_path)


def merge_t_expanded(
    all_source_path: str,
    all_target_path: str,
    all_s_t_labels_path: str,
    all_t_t_labels_path: str,
    source_path: str,
    target_path: str,
    s_t_labels_path: str,
    t_t_labels_path: str,
    source_map_path: str,
    target_map_path: str,
):
    # merge the annotations with the jump sources and targets
    sources = pd.read_parquet(all_source_path)
    targets = pd.read_parquet(all_target_path)
    s_t_labels = pd.read_parquet(all_s_t_labels_path)
    t_t_labels = pd.read_parquet(all_t_t_labels_path)

    source_ids = pd.Series(sources.index)
    target_ids = pd.Series(targets.index)
    s_t_labels = s_t_labels.merge(source_ids, on="Metadata_InChIKey", copy=False)
    s_t_labels = s_t_labels.merge(target_ids, on="Metadata_Symbol", copy=False)

    # gives a list of targets for each source
    src_to_tgt = s_t_labels.groupby("Metadata_InChIKey")["Metadata_Symbol"].apply(
        "|".join
    )

    src_to_tgt = src_to_tgt.str.split("|").explode().reset_index()
    src_to_tgt = src_to_tgt.drop_duplicates()

    # filter sources and targets by ones with annotations (of any kind)
    # for bipartite, we will have 2683 sources and 3780 targets at this point
    sources_with_links = src_to_tgt["Metadata_InChIKey"].unique()
    targets_with_links = src_to_tgt["Metadata_Symbol"].unique()

    more_targets = (
        pd.concat([t_t_labels["target_a"], t_t_labels["target_b"]], axis=0)
        .unique()
        .astype(np.ndarray)
    )
    targets_with_links = np.unique(np.hstack((targets_with_links, more_targets)))

    sources = sources[sources.index.isin(sources_with_links)]
    targets = targets[targets.index.isin(targets_with_links)]

    # map from sources and targets to their ids
    source_map = pd.Series(range(len(sources)), index=sources.index)
    target_map = pd.Series(range(len(targets)), index=targets.index)

    pd.DataFrame(source_map).to_parquet(source_map_path)
    pd.DataFrame(target_map).to_parquet(target_map_path)

    s_t_edges = pd.DataFrame(
        {
            "source": src_to_tgt["Metadata_InChIKey"].map(source_map).values,
            "target": src_to_tgt["Metadata_Symbol"].map(target_map).values,
        }
    )
    t_t_edges = pd.DataFrame(
        {
            "target_a": t_t_labels["target_a"].map(target_map).values,
            "target_b": t_t_labels["target_b"].map(target_map).values,
        }
    )

    sources.to_parquet(source_path)
    targets.to_parquet(target_path)
    s_t_edges.to_parquet(s_t_labels_path)
    t_t_edges.to_parquet(t_t_labels_path)


def merge(
    all_source_path: str,
    all_target_path: str,
    all_s_s_labels_path: str,
    all_s_t_labels_path: str,
    all_t_t_labels_path: str,
    source_path: str,
    target_path: str,
    s_s_labels_path: str,
    s_t_labels_path: str,
    t_t_labels_path: str,
    source_map_path: str,
    target_map_path: str,
):
    # merge the annotations with the jump sources and targets
    sources = pd.read_parquet(all_source_path)
    targets = pd.read_parquet(all_target_path)
    s_s_labels = pd.read_parquet(all_s_s_labels_path)
    s_t_labels = pd.read_parquet(all_s_t_labels_path)
    t_t_labels = pd.read_parquet(all_t_t_labels_path)

    source_ids = pd.Series(sources.index)
    target_ids = pd.Series(targets.index)
    s_t_labels = s_t_labels.merge(source_ids, on="Metadata_InChIKey", copy=False)
    s_t_labels = s_t_labels.merge(target_ids, on="Metadata_Symbol", copy=False)

    # gives a list of targets for each source
    src_to_tgt = s_t_labels.groupby("Metadata_InChIKey")["Metadata_Symbol"].apply(
        "|".join
    )

    src_to_tgt = src_to_tgt.str.split("|").explode().reset_index()
    src_to_tgt = src_to_tgt.drop_duplicates()

    # filter sources and targets by ones with annotations (of any kind)
    # for bipartite, we will have 2683 sources and 3780 targets at this point
    sources_with_links = src_to_tgt["Metadata_InChIKey"].unique()
    targets_with_links = src_to_tgt["Metadata_Symbol"].unique()

    more_sources = (
        pd.concat([s_s_labels["source_a"], s_s_labels["source_b"]], axis=0)
        .unique()
        .astype(np.ndarray)
    )
    sources_with_links = np.unique(np.hstack((sources_with_links, more_sources)))

    more_targets = (
        pd.concat([t_t_labels["target_a"], t_t_labels["target_b"]], axis=0)
        .unique()
        .astype(np.ndarray)
    )
    targets_with_links = np.unique(np.hstack((targets_with_links, more_targets)))

    sources = sources[sources.index.isin(sources_with_links)]
    targets = targets[targets.index.isin(targets_with_links)]

    # map from sources and targets to their ids
    source_map = pd.Series(range(len(sources)), index=sources.index)
    target_map = pd.Series(range(len(targets)), index=targets.index)

    pd.DataFrame(source_map).to_parquet(source_map_path)
    pd.DataFrame(target_map).to_parquet(target_map_path)

    s_t_edges = pd.DataFrame(
        {
            "source": src_to_tgt["Metadata_InChIKey"].map(source_map).values,
            "target": src_to_tgt["Metadata_Symbol"].map(target_map).values,
        }
    )
    s_s_edges = pd.DataFrame(
        {
            "source_a": s_s_labels["source_a"].map(source_map).values,
            "source_b": s_s_labels["source_b"].map(source_map).values,
        }
    )
    t_t_edges = pd.DataFrame(
        {
            "target_a": t_t_labels["target_a"].map(target_map).values,
            "target_b": t_t_labels["target_b"].map(target_map).values,
        }
    )

    sources.to_parquet(source_path)
    targets.to_parquet(target_path)
    s_s_edges.to_parquet(s_s_labels_path)
    s_t_edges.to_parquet(s_t_labels_path)
    t_t_edges.to_parquet(t_t_labels_path)
