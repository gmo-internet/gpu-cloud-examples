# See Also <https://developer.nvidia.com/ja-jp/blog/how-to-use-sft-on-nemo-framework-in-japanese/>
from glob import glob
import json
import os
import random
import pandas as pd

INPUT_PATH = [
    "./data/databricks-dolly-15k-ja/dolly-ja.json",
    "./data/databricks-dolly-15k/dolly.json",
]
OUTPUT_PATH = "./data/sft"
USE_COLS = ["context", "instruction", "response", "source"]
random.seed(42)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Llama 3.1-3.2 templates
INPUT_PROMPT = """<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n"""
NO_INPUT_PROMPT = """<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n"""

def load_dolly_dataset(path, source):
    dataset = pd.read_json(path, lines=True)
    dataset["source"] = source
    return dataset[USE_COLS]

def write_jsonl(fname, json_objs):
    with open(fname, 'wt') as f:
        for o in json_objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

def form_input(row):
    context = row["context"].strip()
    instruction = row["instruction"].strip()
    response = row["response"].strip() + "<|eot_id|>"  # For Llama 3.1-3.2
    assert instruction != ""
    if context != "":
        input = INPUT_PROMPT.format(instruction=instruction, input=context)
    else:
        input = NO_INPUT_PROMPT.format(instruction=instruction)
    return input, response, row["source"]

def prosess(input_path):
    processed = []
    dataset = pd.DataFrame()
    for path in input_path:
        if "dolly.json" in path:
            df = load_dolly_dataset(path, "dolly")
            print("dolly num_records: ", df.shape[0])
        elif "dolly-ja.json" in path:
            df = load_dolly_dataset(path, "dolly-ja")
            print("dolly-ja num_records: ", df.shape[0])
        else:
            print(f"Ignore...: {path}")
        dataset = pd.concat([dataset, df], ignore_index=True)

    # drop duplicated samples
    print("total records: ", dataset.shape[0])
    dataset = dataset[~dataset.duplicated(subset=["context", "instruction", "response"])].reset_index(drop=True)
    print("total records: ", dataset.shape[0])

    for index, row in dataset.iterrows():
        input, output, source = form_input(row)
        processed.append({"input": input, "output": output, "source": source})

    random.shuffle(processed)
    train_ds = processed[:int(len(processed) * 0.9)]
    valid_ds = processed[int(len(processed) * 0.9):]

    write_jsonl(f"{OUTPUT_PATH}/training.jsonl", train_ds)
    write_jsonl(f"{OUTPUT_PATH}/validation.jsonl", valid_ds)

    print("num_train: ", len(train_ds), "num_valid: ", len(valid_ds))
    print(train_ds[0]["input"])
    print(train_ds[0]["output"])
    print(train_ds[0]["source"])

    return

def main():
    prosess(INPUT_PATH)

if __name__ == "__main__":
    main()
