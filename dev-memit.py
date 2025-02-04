from easyeditor import GraceHyperParams, MEMITHyperParams
from easyeditor import BaseEditor

prompts = ["Ray Charles, the", "Grant Hill is a professional", "The law in Ikaalinen declares the language"]
ground_truth = ["piano", "basketball", "Finnish"]
target_new = ["violin", "soccer", "Swedish"]
subject = ["Ray Charles", "Grant Hill", "Ikaalinen"]

hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt2-xl")
hparams.model_name = "/home/zliu/shared_resources/models/gpt/gpt2-xl"

editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=None,
    target_new=target_new,
    subject=subject,
    keep_original_weight=True,
)

print(metrics)
print(type(edited_model))
