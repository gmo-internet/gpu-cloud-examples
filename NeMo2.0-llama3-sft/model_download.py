import nemo_run as run
from nemo.collections import llm

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.LlamaModel(llm.Llama3Config8B()), 
        source="hf://meta-llama/Meta-Llama-3-8B",
        )
if __name__ == "__main__":
    import_ckpt = configure_checkpoint_conversion()
    run.run(import_ckpt)
