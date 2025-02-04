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
    hparams.edit_lr = args.edit_lr

    editor = BaseEditor.from_hparams(hparams)

    examples = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-page.jsonl")

    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")

    edit_metrics = []
    all_results = []

    for ex in tqdm(examples[:], "MEND editing"):
        print("Example ID:", ex["id"])
        prompts = [q["question"] for q in ex["single_hop_efficacy"]]
        target_new = [q["answer"] for q in ex["single_hop_efficacy"]]
        metrics, edited_model, _ = editor.edit(
            prompts=prompts, target_new=target_new, ground_truth=None, sequential_edit=True
        )
        edit_metrics.append({"id": ex["id"], "metrics": metrics})

        generation_config = GenerationConfig(
            do_sample=cfg.generation.do_sample,
            top_k=cfg.generation.top_k,
            top_p=cfg.generation.top_p,
            temperature=cfg.generation.temperature,
            pad_token_id=editor.tok.pad_token_id,
            bos_token_id=editor.tok.bos_token_id,
            eos_token_id=editor.tok.eos_token_id,
            max_new_tokens=cfg.generation.max_new_tokens,
            num_return_sequences=cfg.generation.n_decoding_example,
        )
        for question_type in ["multi_hop_efficacy", "single_hop_efficacy"]:
            inferencer = QAInferencer(
                cfg.evaluator.inferencers[0],
                cfg.seed,
                rag_model=None,
                queries=ex[question_type],
            )
            result_df = eval_inferencer(
                inferencer,
                edited_model,
                tokenizer=editor.tok,
                generation_cfg=generation_config,
            )
            result_df.insert(0, "question_type", question_type)
            result_df.insert(0, "id", ex["id"])
            all_results.append(result_df)
        all_results.append(result_df)

    all_results = pd.concat(all_results)

    io.dump_jsonlines(edit_metrics, f"mend_two-doc_single-edit_eval_lr{hparams.edit_lr}.jsonl")

    all_results.to_excel(
        f"mend_two-doc_single-edit_eval_lr{hparams.edit_lr}.xlsx",
        index=False,
    )


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
