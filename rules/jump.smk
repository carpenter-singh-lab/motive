wildcard_constraints:
    resource=r"^inputs/.*",
    tgt_type=r"orf|crispr",


from motive import jump
from motive import store_splits

S3_ROOT = "s3://staging-cellpainting-gallery/cpg0034-arevalo-su-motive/broad/workspace/publication_data/2024_MOTIVE"


rule download_from_s3:
    output:
        "{resource}",
    shell:
        #f"aws s3 cp --no-sign-request {S3_ROOT}/{{wildcards.resource}} {{wildcards.resource}}"
        f"aws s3 cp {S3_ROOT}/{{wildcards.resource}} {{wildcards.resource}}"


rule prepare_jump_sources:
    input:
        "inputs/compound/features.parquet",
        "inputs/compound/meta.csv.gz",
    output:
        "data/all_source.parquet",
    run:
        jump.load_jump_compounds(*input, *output)


rule prepare_jump_targets:
    input:
        "inputs/{tgt_type}/features.parquet",
        "inputs/{tgt_type}/meta.csv.gz",
    output:
        "data/{tgt_type}_all_target.parquet",
    run:
        jump.load_jump_targets(*input, *output)


rule prepare_jump_labels:
    input:
        "inputs/annotations/compound_gene.parquet",
    output:
        "data/all_s_t_labels.parquet",
    run:
        jump.load_reduced_jump_annotations(*input, *output)


rule prepare_s_s_labels:
    input:
        "inputs/annotations/compound_compound.parquet",
        "data/all_source.parquet",
    output:
        "data/all_s_s_labels.parquet",
    run:
        jump.load_compound_compound_annotations(*input, *output)


rule prepare_t_t_labels:
    input:
        "inputs/annotations/gene_gene.parquet",
        "data/{tgt_type}_all_target.parquet",
    output:
        "data/{tgt_type}_all_t_t_labels.parquet",
    run:
        jump.load_gene_gene_annotations(*input, *output)


rule merge_jump_bipartite:
    input:
        all_source_path="data/all_source.parquet",
        all_target_path="data/{tgt_type}_all_target.parquet",
        all_s_t_labels_path="data/all_s_t_labels.parquet",
    output:
        source_path="data/bipartite/{tgt_type}/source.parquet",
        target_path="data/bipartite/{tgt_type}/target.parquet",
        labels_path="data/bipartite/{tgt_type}/s_t_labels.parquet",
        source_map_path="data/bipartite/{tgt_type}/source_map.parquet",
        target_map_path="data/bipartite/{tgt_type}/target_map.parquet",
    run:
        jump.merge_bipartite(
            input.all_source_path,
            input.all_target_path,
            input.all_s_t_labels_path,
            output.source_path,
            output.target_path,
            output.labels_path,
            output.source_map_path,
            output.target_map_path,
        )


rule merge_jump_s_expanded:
    input:
        all_source_path="data/all_source.parquet",
        all_target_path="data/{tgt_type}_all_target.parquet",
        all_s_s_labels_path="data/all_s_s_labels.parquet",
        all_s_t_labels_path="data/all_s_t_labels.parquet",
    output:
        source_path="data/s_expanded/{tgt_type}/source.parquet",
        target_path="data/s_expanded/{tgt_type}/target.parquet",
        s_s_labels_path="data/s_expanded/{tgt_type}/s_s_labels.parquet",
        s_t_labels_path="data/s_expanded/{tgt_type}/s_t_labels.parquet",
        source_map_path="data/s_expanded/{tgt_type}/source_map.parquet",
        target_map_path="data/s_expanded/{tgt_type}/target_map.parquet",
    run:
        jump.merge_s_expanded(
            input.all_source_path,
            input.all_target_path,
            input.all_s_s_labels_path,
            input.all_s_t_labels_path,
            output.source_path,
            output.target_path,
            output.s_s_labels_path,
            output.s_t_labels_path,
            output.source_map_path,
            output.target_map_path,
        )


rule merge_jump_t_expanded:
    input:
        all_source_path="data/all_source.parquet",
        all_target_path="data/{tgt_type}_all_target.parquet",
        all_s_t_labels_path="data/all_s_t_labels.parquet",
        all_t_t_labels_path="data/{tgt_type}_all_t_t_labels.parquet",
    output:
        source_path="data/t_expanded/{tgt_type}/source.parquet",
        target_path="data/t_expanded/{tgt_type}/target.parquet",
        s_t_labels_path="data/t_expanded/{tgt_type}/s_t_labels.parquet",
        t_t_labels_path="data/t_expanded/{tgt_type}/t_t_labels.parquet",
        source_map_path="data/t_expanded/{tgt_type}/source_map.parquet",
        target_map_path="data/t_expanded/{tgt_type}/target_map.parquet",
    run:
        jump.merge_t_expanded(
            input.all_source_path,
            input.all_target_path,
            input.all_s_t_labels_path,
            input.all_t_t_labels_path,
            output.source_path,
            output.target_path,
            output.s_t_labels_path,
            output.t_t_labels_path,
            output.source_map_path,
            output.target_map_path,
        )


rule merge_jump_big:
    input:
        all_source_path="data/all_source.parquet",
        all_target_path="data/{tgt_type}_all_target.parquet",
        all_s_s_labels_path="data/all_s_s_labels.parquet",
        all_s_t_labels_path="data/all_s_t_labels.parquet",
        all_t_t_labels_path="data/{tgt_type}_all_t_t_labels.parquet",
    output:
        source_path="data/st_expanded/{tgt_type}/source.parquet",
        target_path="data/st_expanded/{tgt_type}/target.parquet",
        s_s_labels_path="data/st_expanded/{tgt_type}/s_s_labels.parquet",
        s_t_labels_path="data/st_expanded/{tgt_type}/s_t_labels.parquet",
        t_t_labels_path="data/st_expanded/{tgt_type}/t_t_labels.parquet",
        source_map_path="data/st_expanded/{tgt_type}/source_map.parquet",
        target_map_path="data/st_expanded/{tgt_type}/target_map.parquet",
    run:
        jump.merge(
            input.all_source_path,
            input.all_target_path,
            input.all_s_s_labels_path,
            input.all_s_t_labels_path,
            input.all_t_t_labels_path,
            output.source_path,
            output.target_path,
            output.s_s_labels_path,
            output.s_t_labels_path,
            output.t_t_labels_path,
            output.source_map_path,
            output.target_map_path,
        )


rule split_bipartite:
    input:
        "data/bipartite/{tgt_type}/s_t_labels.parquet",
    output:
        "data/bipartite/{tgt_type}/{leave_out}/s_t_labels.parquet",
    run:
        store_splits.bipartite(*input, wildcards.leave_out, *output)


rule split_s_expanded:
    input:
        "data/s_expanded/{tgt_type}/s_s_labels.parquet",
        "data/s_expanded/{tgt_type}/s_t_labels.parquet",
    output:
        "data/s_expanded/{tgt_type}/{leave_out}/s_s_labels.parquet",
        "data/s_expanded/{tgt_type}/{leave_out}/s_t_labels.parquet",
    run:
        store_splits.s_expanded(*input, wildcards.leave_out, *output)


rule split_t_expanded:
    input:
        "data/t_expanded/{tgt_type}/s_t_labels.parquet",
        "data/t_expanded/{tgt_type}/t_t_labels.parquet",
    output:
        "data/t_expanded/{tgt_type}/{leave_out}/s_t_labels.parquet",
        "data/t_expanded/{tgt_type}/{leave_out}/t_t_labels.parquet",
    run:
        store_splits.t_expanded(*input, wildcards.leave_out, *output)


rule split_st_expanded:
    input:
        "data/st_expanded/{tgt_type}/s_s_labels.parquet",
        "data/st_expanded/{tgt_type}/s_t_labels.parquet",
        "data/st_expanded/{tgt_type}/t_t_labels.parquet",
    output:
        "data/st_expanded/{tgt_type}/{leave_out}/s_s_labels.parquet",
        "data/st_expanded/{tgt_type}/{leave_out}/s_t_labels.parquet",
        "data/st_expanded/{tgt_type}/{leave_out}/t_t_labels.parquet",
    run:
        store_splits.st_expanded(*input, wildcards.leave_out, *output)
