# databricks/databricks-dolly-15k
import os
from datasets import load_dataset

# llm-jp/databricks-dolly-15k
DATA_DIR = "./data/databricks-dolly-15k"
os.makedirs(DATA_DIR, exist_ok=True)
dataset = load_dataset("databricks/databricks-dolly-15k")
dataset["train"].to_json(f"{DATA_DIR}/dolly.json", force_ascii=False)

# llm-jp/databricks-dolly-15k-ja
DATA_DIR = "./data/databricks-dolly-15k-ja"
os.makedirs(DATA_DIR, exist_ok=True)
dataset = load_dataset("llm-jp/databricks-dolly-15k-ja")
dataset["train"].to_json(f"{DATA_DIR}/dolly-ja.json", force_ascii=False)

print("[INFO] DONE.")
