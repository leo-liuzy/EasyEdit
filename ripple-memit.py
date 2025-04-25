from easyeditor import MEMITHyperParams
from easyeditor import BaseEditor

from knowledge_propagation.utils import io, vars
import numpy as np
import torch
import random
import transformers
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

prompts = ["Ray Charles, the", "Grant Hill is a professional", "The law in Ikaalinen declares the language"]
# ground_truth = ["piano", "basketball", "Finnish"]
target_new = ["violin", "soccer", "Swedish"]
subject = ["Ray Charles", "Grant Hill", "Ikaalinen"]

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/data/users/zliu/mend/models/Llama-3.2-1B-eos-sft",
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

generation_config = GenerationConfig(
    do_sample=False,  # Greedy
    top_k=None,
    top_p=None,
    temperature=None,
    max_new_tokens=30,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/llama3.2-1B-eos-sft-mid-upper")
# hparams.model_name = "/home/zliu/shared_resources/models/gpt/gpt2-xl"

# hparams.mom2_dataset = "ripple_recent"

val_data = io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/test_aug.jsonl")

all_result_df = []

editor = BaseEditor.from_hparams(hparams)

for val in val_data:
    prompts = [val["prompt"]]
    target_new = [val["object"]]
    subjects = [val["subject"]]

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=None,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True,
    )

    outerloop_queries = []
    for k in ["Logical_Generalization", "Compositionality_I", "Compositionality_II", "Subject_Aliasing"]:
        for instance in datum[k]:
            for q in instance["test_queries"]:
                if len(q["answers"]) > 0 and len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0:
                    q["question_type"] = k
                    outerloop_queries.append(q)

    assert len(outerloop_queries) > 0

    locality_queries = []
    for k in ["Relation_Specificity", "Forgetfulness"]:
        for instance in datum[k]:
            for q in instance["test_queries"]:
                if len(q["answers"]) > 0 and len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0:
                    q["question_type"] = k
                    locality_queries.append(q)
    assert len(locality_queries) > 0

    question_types = [
        ("efficacy", outerloop_queries),
        ("specificity", locality_queries),
    ]
    for question_type, questions in question_types:
        logging.info(f"Question type: {question_type}")

        for q_i, question in enumerate(questions):
            answer_candidates = [a["value"] for a in question["answers"]]
            answer = answer_candidates[0]

            post_result_df = get_eval_result(
                question=question["prompt"],
                answer=answer,
                model=edited_model.model,
                tokenizer=tokenizer,
                config=config,
                generation_config=generation_config,
            )
            post_result_df.insert(0, "stage", "post-edit")
            post_result_df.insert(
                0, "edit_input", "\n\n".join(f"[[{tokenizer.decode(s)}]]" for s in sentences_toks["input_ids"])
            )
            post_result_df.insert(0, "relation", f"{question['relation']}")
            post_result_df.insert(0, "question_tag", f"{question_type}_{question['question_type']}")
            post_result_df.insert(0, "question_type", question_type)
            post_result_df.insert(0, "id", str(i))
            # import pdb
            # pdb.set_trace()
            all_datum_result_df.append(post_result_df)

    del edited_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_datum_result_df = pd.concat(all_datum_result_df)
    all_results.append(all_datum_result_df)

all_results = pd.concat(all_results)
