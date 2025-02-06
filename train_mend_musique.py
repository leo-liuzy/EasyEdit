from easyeditor import BaseEditor
from easyeditor import (
    MENDTrainingHparams,
    MENDHyperParams,
)
from easyeditor import ZsreDataset
from easyeditor import EditTrainer
import typing
import json
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer
from knowledge_propagation.utils import vars, io


class MusiqueDataset(ZsreDataset):
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir

        if config is not None:
            self.config = config
        if config is not None and hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            self.max_length = 40

        # For Meta Training
        if config is not None and hasattr(config, "tokenizer_name"):
            tok_name = config.tokenizer_name if config.tokenizer_name is not None else config.model.name
            tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
            # tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
            #     tok_name, trust_remote_code=True
            # )
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
                print("GPTTokenizer Detected, Set pad token id and left padding!!!")
            elif isinstance(tokenizer, LlamaTokenizer) or "Llama" in tok_name:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
                print("LlamaTokenizer Detected, Set pad token id and left padding!!!")
            elif "qwen" in config.model_name.lower():
                tokenizer.eos_token = "<|endoftext|>"
                tokenizer.pad_token = "<|endoftext|>"
                tokenizer.unk_token = "<|endoftext|>"
                # tokenizer.padding_side = 'left'
                # print('QwenTokenizer Detected, Set pad token id and left padding!!!')
            elif "mistral" in config.model_name.lower():
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
                print("MistralTokenizer Detected, Set pad token id and left padding!!!")
            self.tok = tokenizer

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            assert "nq question: " in record["loc"], f"Neighborhood prompt missing `nq question:`. Check for errors?"
            # ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            if record["alt"] == "":
                continue
            data.append(
                {
                    "case_id": i,
                    "prompt": record["src"],
                    "target_new": record["alt"],
                    "ground_truth": record["answers"][0],
                    "rephrase_prompt": record["rephrase"],
                    # "neighborhood_prompts": [
                    #     {
                    #         "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                    #         "target": tok.decode(ans_toks[i]),
                    #     }
                    #     for i in range(len(ans_toks))
                    # ],
                    "locality_prompt": record["loc"],
                    "locality_ground_truth": record["loc_ans"],
                    "cond": "{} >> {} || {}".format(
                        record["answers"][0],
                        record["alt"],
                        record["src"],
                    ),
                }
            )

        if size is not None:
            data = data[:size]
        self._data = data


def main(args):
    training_hparams = MENDTrainingHparams.from_hparams(args.config_path)
    # training_hparams = MENDTrainingHparams.from_hparams('./hparams/MEND/llama3.2-3B.yaml')
    train_ds = MusiqueDataset(f"{vars.DATA_DIR}/musique_mend/musique_ans_v1.0_train.jsonl", config=training_hparams)
    eval_ds = MusiqueDataset(f"{vars.DATA_DIR}/musique_mend/musique_ans_v1.0_dev.jsonl", config=training_hparams)
    # trainer = EditTrainer(config=training_hparams, train_set=train_ds, val_set=eval_ds)

    # trainer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    main(args)
