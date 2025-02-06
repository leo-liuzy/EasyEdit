import hydra

import pandas as pd

from easyeditor import GraceHyperParams, MEMITHyperParams, MENDHyperParams
from easyeditor import BaseEditor, EditTrainer, MendRewriteExecutor
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

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import torch


def load_llama_model_tokenzier(hparams):
    model_name = hparams.model_name

    device_map = "auto" if hparams.model_parallel else None
    torch_dtype = torch.float16 if hasattr(hparams, "fp16") and hparams.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if hparams.model_parallel:
        hparams.device = str(model.device).split(":")[1]
    if not hparams.model_parallel and hasattr(hparams, "device"):
        model.to(f"cuda:{hparams.device}")
    return model, tokenizer


def main(args):
    hparams = MENDHyperParams.from_hparams("hparams/MEND/llama3.2-1B.yaml")
    hparams.val_batch_size = 1
    train_ds = ZsreDataset("./data/zsre/zsre_mend_train.json", config=hparams)
    eval_ds = ZsreDataset("./data/zsre/zsre_mend_eval.json", size=10, config=hparams)
    # hparams.edit_lr = args.edit_lr
    model, tokenizer = load_llama_model_tokenzier(hparams)
    mend_rewriter = MendRewriteExecutor()
    mend_rewriter.init_model(model, tokenizer, hparams)

    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")

    generation_config = GenerationConfig(
        do_sample=cfg.generation.do_sample,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        temperature=cfg.generation.temperature,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=cfg.generation.max_new_tokens,
        num_return_sequences=cfg.generation.n_decoding_example,
    )
    all_results = []

    # print(f"Model checksum [init]: {sum(p.sum() for p in editor.model.parameters())}")  # if p.requires_grad
    # print(
    #     f"Model checksum [init; requires_grad]: {sum(p.sum() for p in editor.model.parameters() if p.requires_grad)}"
    # )  # if p.requires_grad
    # print(f"no_grad_params: {[n for n, p in editor.model.named_parameters() if not p.requires_grad]}")
    # print(f"grad_params: {[n for n, p in editor.model.named_parameters() if p.requires_grad]}")
    # # editor.model.eval()
    # editor.model.train()
    # print(f"Model checksum [init after eval]: {sum(p.sum() for p in editor.model.parameters())}")  # if p.requires_grad
    # print(
    #     f"Model checksum [init after eval; requires_grad]: {sum(p.sum() for p in editor.model.parameters() if p.requires_grad)}"
    # )  # if p.requires_grad
    for i, d in enumerate(eval_ds):
        prompts = [d["prompt"]]
        target_new = [d["target_new"]]

        requests = _prepare_requests(
            prompts,
            target_new,
            ground_truth=["<|endoftext|>"] * len(prompts),
        )
        targets = [(" " if request["target_new"][0] != " " else "") + request["target_new"] for request in requests]
        sentences = [request["prompt"] + targets[i] for i, request in enumerate(requests)]

        # Tokenize
        sent_tok = tokenizer(sentences, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
        target_tok = tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False).to(
            f"cuda:{hparams.device}"
        )
        edit_inner = dict(
            input_ids=sent_tok["input_ids"],
            attention_mask=sent_tok["attention_mask"],
            labels=target_tok["input_ids"],
        )
        edited_model, metrics = mend_rewriter.edit_step(
            dict(edit_inner=edit_inner), training=False, update_base_model=True
        )
        print(
            f"[{i}] edit/acc:",
            metrics["pre_edit/acc"],
            "-->",
            metrics["post_edit/acc"],
        )
        print(
            f"[{i}] target:",
            "\n",
            metrics["pre_edit/target"],
        )
        print(
            f"[{i}] edit/gen:",
            "\n",
            metrics["pre_edit/gen"],
            "\n",
            "-->",
            "\n",
            metrics["post_edit/gen"],
        )

        # inferencer = QAInferencer(
        #     cfg.evaluator.inferencers[0],
        #     cfg.seed,
        #     rag_model=None,
        #     queries=[{"question": d["prompt"], "answer": d["target_new"]}],
        # )

        # result_df_post_edit = eval_inferencer(
        #     inferencer,
        #     edited_model.model,
        #     tokenizer=editor.tok,
        #     generation_cfg=generation_config,
        # )

        # result_df_pre_edit = eval_inferencer(
        #     inferencer,
        #     mend_rewriter.model,
        #     tokenizer=editor.tok,
        #     generation_cfg=generation_config,
        # )
        print("=" * 20)
        # result_df_pre_edit.insert(0, "stage", "pre-edit")
        # result_df_post_edit.insert(0, "stage", "post-edit")
        # all_results.append(result_df_pre_edit)
        # all_results.append(result_df_post_edit)
    # all_results = pd.concat(all_results)

    # all_results.to_excel(
    #     f"mend_zsRE_eval.xlsx",
    #     index=False,
    # )
    # print(
    #     f"[{i}] edit/acc:",
    #     metrics["pre_edit/acc"],
    #     "-->",
    #     metrics["post_edit/acc"],
    # )
    # print(
    #     f"[{i}] target:",
    #     metrics["pre_edit/target"],
    # )
    # print(
    #     f"[{i}] edit/gen:",
    #     metrics["pre_edit/gen"],
    #     "-->",
    #     metrics["post_edit/gen"],
    # )
    # print("=" * 20)
    # print(f"[{i}]", metrics["edit/acc"])
    # print(info_dict)
    # print(
    #     "rewrite_acc:",
    #     metrics[0]["pre"]["rewrite_acc"],
    #     "--> ",
    #     metrics[0]["post"]["rewrite_acc"],
    # )

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
