import hydra

import pandas as pd

from easyeditor import GraceHyperParams, MEMITHyperParams, MENDHyperParams
from easyeditor import BaseEditor, MendRewriteExecutor
from tqdm import tqdm

from knowledge_propagation.utils import vars, io
from knowledge_propagation.modules.inferencers import QAInferencer
from experiments.musique.inference_only import eval_inferencer, macro_averaging

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import torch
import gc


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


class EditLossType(StrEnum):
    sft = "p(y|x)"

    clm = "p(x)"


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
    # hparams.edit_lr = args.edit_lr

    # examples = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-page.jsonl")
    examples = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-paragraph.jsonl")

    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")

    edit_metrics = []
    norm_tracker = []
    all_results = []

    for i, ex in tqdm(enumerate(examples[:]), "MEND editing", total=len(examples)):
        print("Example ID:", ex["id"])

        model, tokenizer = load_llama_model_tokenzier(hparams)
        norm_tracker.append(
            {
                "id": ex["id"],
                "after_n_edit": 0,
                **{n: p.norm().item() for n, p in model.named_parameters() if n in hparams.inner_params},
            }
        )
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
        mend_rewriter = MendRewriteExecutor()
        mend_rewriter.init_model(model, tokenizer, hparams)
        for q_i, q in enumerate(ex["single_hop_efficacy"]):
            if args.edit_loss == EditLossType.sft:
                prompts = [q["question"]]
                target_news = [q["answer"]]

                targets = [(" " if target_new[0] != " " else "") + target_new for target_new in target_news]
                sentences = [prompt + targets[i] for i, prompt in enumerate(prompts)]
            else:
                assert len(q["supporting_text_ids"]) == 1
                sentences = targets = [ex["texts"][q["supporting_text_ids"][0]]]

            sent_tok = tokenizer(sentences, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
            target_tok = tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False).to(
                f"cuda:{hparams.device}"
            )  # add_special_tokens=False to avoid calculating loss with <bos> token among labels
            assert sent_tok["input_ids"].shape[-1] > target_tok["input_ids"].shape[-1]
            edit_inner = dict(
                input_ids=sent_tok["input_ids"],
                attention_mask=sent_tok["attention_mask"],
                labels=target_tok["input_ids"],
            )

            edited_model, metrics = mend_rewriter.edit_step(
                dict(edit_inner=edit_inner), training=False, update_base_model=True
            )
            if "factors" in metrics:
                del metrics["factors"]
            edit_metrics.append({"id": ex["id"], "metrics": metrics})
            print(
                f"[{i}; {q_i}] edit/acc:",
                metrics["pre_edit/acc"],
                "-->",
                metrics["post_edit/acc"],
            )
            print(
                f"[{i}; {q_i}] target:",
                "\n",
                metrics["pre_edit/target"],
            )
            print(
                f"[{i}; {q_i}] edit/gen:",
                "\n",
                metrics["pre_edit/gen"],
                "\n",
                "-->",
                "\n",
                metrics["post_edit/gen"],
            )
            norm_tracker.append(
                {
                    "id": ex["id"],
                    "after_n_edit": q_i + 1,
                    **{
                        n: p.norm().item()
                        for n, p in mend_rewriter.model.named_parameters()
                        if n in hparams.inner_params
                    },
                }
            )
        mend_rewriter.reset_model()
        for question_type in ["multi_hop_efficacy", "single_hop_efficacy"]:
            inferencer = QAInferencer(
                cfg.evaluator.inferencers[0],
                cfg.seed,
                rag_model=None,
                queries=ex[question_type],
            )
            result_df = eval_inferencer(
                inferencer,
                edited_model.model,
                tokenizer=tokenizer,
                generation_cfg=generation_config,
            )
            result_df.insert(0, "question_type", question_type)
            result_df.insert(0, "id", ex["id"])
            all_results.append(result_df)

        del mend_rewriter
        del edited_model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = pd.concat(all_results)

    io.dump_jsonlines(edit_metrics, f"mend_two-doc_single-edit_metrics_{args.edit_loss}.jsonl")
    io.dump_jsonlines(norm_tracker, f"mend_two-doc_single-edit_norms_{args.edit_loss}.jsonl")

    all_results.to_excel(
        f"mend_two-doc_single-edit_eval_{args.edit_loss}.xlsx",
        index=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--edit_loss",
        type=EditLossType,
        # choices=["y|x", "x"],
        required=True,
    )
    args = parser.parse_args()
    main(args)
