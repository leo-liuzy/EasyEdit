from easyeditor import BaseEditor
from easyeditor import (
    MENDTrainingHparams,
    MENDHyperParams,
)
from easyeditor import ZsreDataset
from easyeditor import EditTrainer

training_hparams = MENDTrainingHparams.from_hparams(args.config_path)
# training_hparams = MENDTrainingHparams.from_hparams('./hparams/MEND/llama3.2-3B.yaml')
train_ds = ZsreDataset("./data/zsre/zsre_mend_train.json", config=training_hparams)
eval_ds = ZsreDataset("./data/zsre/zsre_mend_eval.json", config=training_hparams)
trainer = EditTrainer(config=training_hparams, train_set=train_ds, val_set=eval_ds)

trainer.run()
