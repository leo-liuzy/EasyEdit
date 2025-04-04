import hydra

import pandas as pd

from easyeditor import GraceHyperParams, MEMITHyperParams, MENDHyperParams
from easyeditor import BaseEditor, EditTrainer
from easyeditor import ZsreDataset
from easyeditor.editors.utils import _prepare_requests
from tqdm import tqdm
from copy import deepcopy

from knowledge_propagation.utils import vars, io
from knowledge_propagation.modules.inferencers import QAInferencer
from experiments.musique.inference_only import eval_inferencer, macro_averaging

from transformers import AutoTokenizer, GenerationConfig


import hydra

import pandas as pd

from easyeditor import GraceHyperParams, MEMITHyperParams, MENDHyperParams
from easyeditor import BaseEditor
from tqdm import tqdm

from knowledge_propagation.utils import vars, io
from knowledge_propagation.modules.inferencers import QAInferencer
from experiments.musique.inference_only import eval_inferencer, macro_averaging

from transformers import AutoTokenizer, GenerationConfig


def main(args):
    hparams = MENDHyperParams.from_hparams("hparams/MEND/llama3.2-1B.yaml")
    hparams.val_batch_size = 1
    train_ds = ZsreDataset("./data/zsre/zsre_mend_train.json", config=hparams)
    eval_ds = ZsreDataset("./data/zsre/zsre_mend_eval.json", size=10, config=hparams)
    trainer = EditTrainer(config=hparams, train_set=train_ds, val_set=eval_ds)
    # trainer.validate(log=True)

    for i, d in enumerate(eval_ds):
        prompts = [d["prompt"]]
        target_new = [d["target_new"]]

        requests = _prepare_requests(
            prompts,
            target_new,
            ground_truth=["<|endoftext|>"] * len(prompts),
        )
        targets = [request["target_new"] for request in requests]
        sentences = [
            request["prompt"] + (" " if request["target_new"][0] != " " else "") + targets[i]
            for i, request in enumerate(requests)
        ]

        # Tokenize
        sent_tok = eval_ds.tok(sentences, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
        target_tok = eval_ds.tok(targets, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
        edit_inner = dict(
            input_ids=sent_tok["input_ids"],
            attention_mask=sent_tok["attention_mask"],
            labels=target_tok["input_ids"],
        )
        edited_model, model_info, metrics = trainer.simple_edit(batch=dict(edit_inner=edit_inner), training=False)
        print(
            f"[{i}] edit/acc:",
            metrics["pre_edit/acc"],
            "-->",
            metrics["post_edit/acc"],
        )
        print(
            f"[{i}] target:",
            metrics["pre_edit/target"],
        )
        print(
            f"[{i}] edit/gen:",
            metrics["pre_edit/gen"],
            "-->",
            metrics["post_edit/gen"],
        )
        print("=" * 20)

    # hparams.edit_lr = args.edit_lr

    # editor = BaseEditor.from_hparams(hparams)

    # examples = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-page.jsonl")

    # with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
    #     cfg = hydra.compose(config_name="fft.yaml")

    # edit_metrics = []
    # all_results = []

    # ex = examples[0]

    # prompts = [q["question"] for q in ex["single_hop_efficacy"]]
    # target_new = [q["answer"] for q in ex["single_hop_efficacy"]]

    # requests = _prepare_requests(
    #     prompts,
    #     target_new,
    #     ground_truth=["<|endoftext|>"] * len(prompts),
    # )

    # targets = [(" " if request["target_new"][0] != " " else "") + request["target_new"] for request in requests]
    # sentences = [request["prompt"] + targets[i] for i, request in enumerate(requests)]

    # # Tokenize
    # sent_tok = editor.tok(sentences, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
    # target_tok = editor.tok(targets, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")

    # # Define labels
    # label_tok = deepcopy(sent_tok["input_ids"])
    # for i in range(label_tok.size(0)):
    #     target_len = target_tok["attention_mask"][i].sum()
    #     padding_len = sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
    #     label_tok[i][: -target_len - padding_len] = -100
    #     label_tok[i][label_tok[i] == editor.tok.pad_token_id] = -100

    # # Run MEND
    # edit_inner = dict(
    #     input_ids=sent_tok["input_ids"],
    #     attention_mask=sent_tok["attention_mask"],
    #     labels=target_tok["input_ids"],
    # )
    # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}

    # editor.model.eval()
    # edited_model, model_info = editor.model.edit(edit_inner, cond, return_factors=True)

    # metrics, edited_model, _ = editor.edit(
    #     prompts=prompts, target_new=target_new, ground_truth=None, sequential_edit=True
    # )
    # edit_metrics.append({"id": ex["id"], "metrics": metrics})

    # generation_config = GenerationConfig(
    #     do_sample=cfg.generation.do_sample,
    #     top_k=cfg.generation.top_k,
    #     top_p=cfg.generation.top_p,
    #     temperature=cfg.generation.temperature,
    #     pad_token_id=editor.tok.pad_token_id,
    #     bos_token_id=editor.tok.bos_token_id,
    #     eos_token_id=editor.tok.eos_token_id,
    #     max_new_tokens=cfg.generation.max_new_tokens,
    #     num_return_sequences=cfg.generation.n_decoding_example,
    # )
    # for question_type in ["multi_hop_efficacy", "single_hop_efficacy"]:
    #     inferencer = QAInferencer(
    #         cfg.evaluator.inferencers[0],
    #         cfg.seed,
    #         rag_model=None,
    #         queries=ex[question_type],
    #     )
    #     result_df = eval_inferencer(
    #         inferencer,
    #         edited_model,
    #         tokenizer=editor.tok,
    #         generation_cfg=generation_config,
    #     )
    #     result_df.insert(0, "question_type", question_type)
    #     result_df.insert(0, "id", ex["id"])
    #     all_results.append(result_df)

    # all_results.append(result_df)

    # all_results = pd.concat(all_results)

    # io.dump_jsonlines(edit_metrics, f"mend_two-doc_single-edit_eval_lr{hparams.edit_lr}.jsonl")

    # all_results.to_excel(
    #     f"mend_two-doc_single-edit_eval_lr{hparams.edit_lr}.xlsx",
    #     index=False,
    # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--edit_lr",
        type=float,
        required=False,
    )
    args = parser.parse_args()
    main(args)
