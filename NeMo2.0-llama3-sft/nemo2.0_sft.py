import os
import nemo_run as run
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.gpt.data import FineTuningDataModule
import datetime

if __name__ == "__main__":
    seq_length = 2048
    devices = int(os.getenv('SLURM_GPUS_PER_NODE'))
    num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
    micro_batch_size = 4
    global_batch_size = devices * num_nodes * micro_batch_size

    recipe = llm.recipes.llama3_8b.finetune_recipe(
        num_nodes=num_nodes,
        num_gpus_per_node=devices,
    )

    recipe.trainer.val_check_interval = 50
    recipe.trainer.max_steps = 100  # 適切なステップ数に調整

    recipe.data = run.Config(
        FineTuningDataModule,
        dataset_root="/workspace/data/sft",
        seq_length=seq_length,
        tokenizer="meta-llama/Llama-3.1-8B",
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
    )

    run.run(recipe)
