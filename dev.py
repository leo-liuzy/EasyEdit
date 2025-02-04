from easyeditor import GraceHyperParams, MEMITHyperParams
from easyeditor import BaseEditor

prompts = [
    "Which family does Ramalinaceae belong to",
    "What role does Denny Herzig play in football?",
    "Who was the designer of Lahti Town Hall?",
    "What is the original channel that It's a Business played on?",
    "What city did Marl Young live when he died?",
    "Steve Jobs was the founder of",
    "LeBron James plays the sport of",
    "The manufacturer of Colt King Cobra was who",
]
ground_truth = [
    "Lecanorales",
    "defender",
    "Eliel Saarinen",
    "DuMont Television Network",
    "Los Angeles",
    "Apple",
    "basketball",
    "Colt's Manufacturing Company",
]
target_new = [
    "Lamiinae",
    "winger",
    "Alfred Lahti",
    "ITV",
    "New Orleans",
    "Microsoft",
    "football",
    "Colt's Manufacturing Corporation",
]

# hparams = GraceHyperParams.from_hparams("./hparams/GRACE/gpt2-xl")
# hparams.model_name = "/home/zliu/shared_resources/models/gpt/gpt2-xl"
# editor = BaseEditor.from_hparams(hparams)
# metrics, edited_model, _ = editor.edit(
#     prompts=prompts, ground_truth=None, target_new=target_new, keep_original_weight=True
# )

hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt2-xl")
hparams.model_name = "/home/zliu/shared_resources/models/gpt/gpt2-xl"

editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=None,
    target_new=target_new,
    # subject=subject,
    # locality_inputs=locality_inputs,
    # portability_inputs=portability_inputs,
    keep_original_weight=True,
)

print(metrics)
print(type(edited_model))
