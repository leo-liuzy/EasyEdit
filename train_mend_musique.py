from easyeditor import BaseEditor
from easyeditor import (
    MENDTrainingHparams,
    MENDHyperParams,
)
from easyeditor import ZsreDataset
from easyeditor import EditTrainer
from easyeditor.trainer.utils import dict_to
import typing
import json
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer
from knowledge_propagation.utils import vars, io


from enum import Enum


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class DocumentCondition(StrEnum):
    two_doc = "two"

    two_doc_seq = "two_seq"


class MetaTrainLoss(StrEnum):
    sft = "p(y|x)"

    clm = "p(x)"


class MusiqueDataset(ZsreDataset):
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = None,
        config=None,
        edit_loss: MetaTrainLoss = MetaTrainLoss.clm,
        doc_condition: DocumentCondition = DocumentCondition.two_doc,
        *args,
        **kwargs,
    ):
        musique_loc = str(Path(data_dir))

        if config is not None:
            self.config = config
        # if config is not None and hasattr(config, "max_length"):
        #     self.max_length = config.max_length
        # else:
        #     self.max_length = 40
        self.edit_loss = edit_loss
        self.doc_condition = doc_condition
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

        raw = io.load_jsonlines(musique_loc)

        if "_train" in musique_loc:
            zsRE_raw = io.load_json("/data/users/zliu/EasyEdit/data/zsre/zsre_mend_train.json")
        else:
            "_dev" in musique_loc
            zsRE_raw = io.load_json("/data/users/zliu/EasyEdit/data/zsre/zsre_mend_eval.json")

        assert len(zsRE_raw) >= len(raw)

        data = []
        for i, (record, zsre_record) in enumerate(zip(raw, zsRE_raw)):
            assert (
                "nq question: " in zsre_record["loc"]
            ), "Neighborhood prompt missing `nq question:`. Check for errors?"
            # ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            # if record["alt"] == "":
            #     continue
            prompt_target_pair = []

            if self.edit_loss == MetaTrainLoss.clm:
                if self.doc_condition == DocumentCondition.two_doc_seq:
                    texts = record["texts"]
                else:
                    texts = ["\n\n".join(record["texts"])]
                prompt_target_pair = [(t, t) for t in texts]
            else:
                # print("Only using multi-hop question for SFT loss so far")
                prompt_target_pair = [(q["question"], q["answer"]) for q in record["multi_hop_efficacy"]]

            # if self.edit_loss == MetaTrainLoss.clm:
            for t_i, (prompt, target) in enumerate(prompt_target_pair):
                data.append(
                    {
                        "case_id": f"{i}-t{t_i}",
                        "prompt": prompt,
                        "target_new": target,
                        "locality_prompt": zsre_record["loc"],
                        "locality_ground_truth": zsre_record["loc_ans"],
                    }
                )

        if size is not None:
            data = data[:size]
        self._data = data

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [(" " if b["target_new"][0] != " " else "") + b["target_new"] for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [(" " if b["locality_ground_truth"][0] != " " else "") + b["locality_ground_truth"] for b in batch]
        if self.edit_loss == MetaTrainLoss.sft:
            src = [src_ + trg_ for src_, trg_ in zip(src, trg)]
        # rephrase = [rephrase_ + " " + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        loc = [loc_ + " " + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        if "gpt" in self.config.tokenizer_class.lower():
            trg = [" " + t for t in trg]
            loc_ans = [" " + t for t in loc_ans]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                # max_length=self.max_length,
                # truncation=True,
                add_special_tokens=k1 == "src",
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)


def main(args):
    training_hparams = MENDTrainingHparams.from_hparams(args.config_path)
    # training_hparams = MENDTrainingHparams.from_hparams('./hparams/MEND/llama3.2-3B.yaml')
    # train_ds = ZsreDataset("./data/zsre/zsre_mend_train.json", config=training_hparams)
    train_ds = MusiqueDataset(
        f"{vars.DATA_DIR}/musique_mend/musique_ans_v1.0_train.jsonl",
        config=training_hparams,
        edit_loss=args.edit_loss,
        doc_condition=args.doc_condition,
    )
    eval_ds = MusiqueDataset(
        f"{vars.DATA_DIR}/musique_mend/musique_ans_v1.0_dev.jsonl",
        config=training_hparams,
        edit_loss=args.edit_loss,
        doc_condition=args.doc_condition,
    )
    trainer = EditTrainer(config=training_hparams, train_set=train_ds, val_set=eval_ds)

    trainer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--edit_loss",
        type=MetaTrainLoss,
        required=True,
    )
    parser.add_argument(
        "--doc_condition",
        type=DocumentCondition,
        required=True,
    )
    args = parser.parse_args()
    main(args)
