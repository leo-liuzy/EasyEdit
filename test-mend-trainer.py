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
    hparams.archive = args.model_archive
    train_ds = ZsreDataset("./data/zsre/zsre_mend_train.json", config=hparams)
    eval_ds = ZsreDataset("./data/zsre/zsre_mend_eval.json", config=hparams)
    trainer = EditTrainer(config=hparams, train_set=train_ds, val_set=eval_ds)
    trainer.validate(log=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--model_archive",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
