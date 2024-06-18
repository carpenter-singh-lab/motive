include: "rules/jump.smk"


TGT_TYPE = ["orf", "crispr"]
SPLIT = ["random", "source", "target"]
GRAPH_TYPE = ["bipartite", "s_expanded", "t_expanded", "st_expanded"]


rule all:
    input:
        expand(
            "data/{graph_type}/{tgt_type}/{split}/s_t_labels.parquet",
            tgt_type=TGT_TYPE,
            graph_type=GRAPH_TYPE,
            split=SPLIT,
        ),
