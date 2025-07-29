# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}

DATA_PATH = "/home/colligo/project/vlm/Bagel/datasets/"
DATA_PATH2 = "/home/colligo/project/vlm/FusionBench"

DATA_PATH = "/home/colligo/project/vlm/FusionBench/src/train/bagel/datasets"


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': f'{DATA_PATH}/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': f'{DATA_PATH}/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": f'{DATA_PATH}/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
        'visual_jigsaw_generation': {
            'data_dir': f'{DATA_PATH2}/data/train/visual_jigsaw_generation',
            'num_files': 160,
            'num_total_samples': 160000,
            'parquet_info_path': f'{DATA_PATH2}/data/train/visual_jigsaw_generation/parquet_info/visual_jigsaw_generation.json'
        },
        'visual_jigsaw_3x3_generation': {
            'data_dir': f'{DATA_PATH2}/src/fusionbench/image/visual_jigsaw_3x3_train/problems/generation_parquet',
            'num_files': 40,  
            'num_total_samples': 40000,
            'parquet_info_path': f'{DATA_PATH2}/src/fusionbench/image/visual_jigsaw_3x3_train/problems/generation_parquet/parquet_info/visual_jigsaw_3x3_generation.json'
        },
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': f'{DATA_PATH}/bagel_example/vlm/images',
			'jsonl_path': f'{DATA_PATH}/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
        'visual_jigsaw_position': {
            'data_dir': f'',
            'jsonl_path': f'{DATA_PATH2}/data/train/visual_jigsaw_position/problems/visual_jigsaw_position_train_vlm.jsonl',
            'num_total_samples': 160000
        },
        'visual_jigsaw_mapping': {
            'data_dir': f'',
            'jsonl_path': f'{DATA_PATH2}/src/fusionbench/image/visual_jigsaw_position_train/problems/visual_jigsaw_mapping_train_bagel_vlm.jsonl',
            'num_total_samples': 160000
        },
        'visual_jigsaw_3x3_position': {
            'data_dir': f'',
            'jsonl_path': f'{DATA_PATH2}/src/fusionbench/image/visual_jigsaw_3x3_train/problems/visual_jigsaw_3x3_position_train_bagel_vlm.jsonl',
            'num_total_samples': 360000
        },
        'visual_jigsaw_3x3_mapping': {
            'data_dir': f'',
            'jsonl_path': f'{DATA_PATH2}/src/fusionbench/image/visual_jigsaw_3x3_train/problems/visual_jigsaw_3x3_mapping_train_bagel_vlm.jsonl',
            'num_total_samples': 40000
        },
    },
}