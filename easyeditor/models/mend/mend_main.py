import os
from copy import deepcopy
from typing import Dict, List

import hydra
import torch
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
import time
from easyeditor.trainer.utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from easyeditor.trainer.losses import kl_loc_loss

from ...trainer import MEND
from .mend_hparams import MENDHyperParams
from .mend_multimodal_hparams import MENDMultimodalHparams


class MendRewriteExecutor:
    def __init__(self):
        self.is_init = False

    def init_model(self, model, tok, params: MENDHyperParams):
        assert params.archive is not None or print(f"Training weights Needed....")

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        self.config = params
        # add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        self.alg = MEND(self.model, params, lambda: deepcopy(self.model))
        d = torch.load(params.archive, map_location="cpu")

        self.alg.load_state_dict({k.replace("gtn.", "mend."): v for k, v in d["model"].items()})
        # if params.model_parallel:
        self.alg.mend.to(deque(self.alg.model.parameters(), maxlen=1)[0].device)
        # else:
        #     self.alg.to(torch.device(f'cuda:{params.device}'))

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in params.inner_params:
                p.requires_grad = False
        self.is_init = True

    def reset_model(self):
        self.is_init = False
        del self.model, self.tokenizer, self.alg

    def edit_step(self, batch, training: bool, update_base_model: bool = False):
        self.alg.train(training)
        self.alg.model.train(training)

        # with torch.no_grad():
        #     base_logits = self.alg(**batch["loc"])

        with torch.set_grad_enabled(training):
            # Editing loss
            pre_edit_logits = self.alg(**batch["edit_inner"])
            targ = batch["edit_inner"]["labels"]
            pre_edit_target_logits = pre_edit_logits[:, :-1]
            pre_edit_target_logits = pre_edit_target_logits[:, -targ.size(1) :]
            pre_edit_gen = [self.tokenizer.decode(p) for p in pre_edit_target_logits.argmax(-1)]
        # Do the edit
        start = time.time()
        edited_model, model_info = self.alg.edit(batch["edit_inner"], return_factors=True)

        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_logits = edited_model(**batch["edit_inner"])
            targ = batch["edit_inner"]["labels"]
            target_logits = post_edit_logits[:, :-1]
            target_logits = target_logits[:, -targ.size(1) :]

            post_edit_gen = [self.tokenizer.decode(p) for p in target_logits.argmax(-1)]

        with torch.no_grad():
            post_edit_dict = self.alg.edit_loss_fn(self.config, post_edit_logits, batch["edit_inner"]["labels"])
            pre_edit_dict = self.alg.edit_loss_fn(self.config, pre_edit_logits, batch["edit_inner"]["labels"])

        info_dict = {}
        info_dict["post_edit/acc"] = post_edit_dict["acc"].item()
        info_dict["post_edit/log_prob"] = post_edit_dict["log_prob"].item()
        info_dict["post_edit/prob"] = post_edit_dict["prob"].item()
        info_dict["post_edit/target"] = [self.tokenizer.decode(t) for t in targ]
        info_dict["post_edit/gen"] = post_edit_gen

        info_dict["pre_edit/acc"] = pre_edit_dict["acc"].item()
        info_dict["pre_edit/log_prob"] = pre_edit_dict["log_prob"].item()
        info_dict["pre_edit/prob"] = pre_edit_dict["prob"].item()
        info_dict["pre_edit/target"] = [self.tokenizer.decode(t) for t in targ]
        info_dict["pre_edit/gen"] = pre_edit_gen
        info_dict["time/edit"] = edit_time

        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}
        print(f"Base model weight checksum: {sum(p.sum() for p in self.alg.model.parameters())}")

        if not update_base_model:
            return edited_model, info_dict

        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()

        # Edit!
        torch_factors = {k: torch.tensor(v) for k, v in factors.items()}
        # weights_copy = {}
        with torch.no_grad():
            for n, p in self.alg.model.named_parameters():
                uname, _ = f"{n}.u", f"{n}.v"
                if uname in torch_factors:
                    with torch.no_grad():
                        p.copy_(edited_model.model.state_dict()[n])
        return edited_model, info_dict

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MENDHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs,
    ):
        """
        Given a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns a dictionary of numpy arrays that specifies
        how mend will change the weights of the model.
        """

        if not self.is_init:
            self.init_model(model, tok, hparams)

        weights_copy = {}
        model = deepcopy(self.model) if copy else self.model

        # Define i/o
        # targets = [(" " if request["target_new"][0] != " " else "") + request["target_new"] for request in requests]
        targets = [request["target_new"] for request in requests]
        sentences = [
            request["prompt"] + (" " if request["target_new"][0] != " " else "") + targets[i]
            for i, request in enumerate(requests)
        ]

        # Tokenize
        sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
        target_tok = self.tokenizer(targets, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")

        # Define labels
        label_tok = deepcopy(sent_tok["input_ids"])
        for i in range(label_tok.size(0)):
            target_len = target_tok["attention_mask"][i].sum()
            padding_len = sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
            label_tok[i][: -target_len - padding_len] = -100
            label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

        # Run MEND
        edit_inner = dict(
            input_ids=sent_tok["input_ids"],
            attention_mask=sent_tok["attention_mask"],
            labels=target_tok["input_ids"],
        )
        cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}

        self.alg.eval()
        # print("edit_inner['input_ids']", self.tokenizer.decode(edit_inner["input_ids"][0]))
        # print(edit_inner["input_ids"][0])
        # print("edit_inner['labels']", self.tokenizer.decode(edit_inner["labels"][0]))
        # print(edit_inner["labels"][0])
        # # print("cond['input_ids']", self.tokenizer.decode(cond["input_ids"][0]))
        # # print("edit_inner", edit_inner)
        # # print("cond", cond)
        # print("edit_inner")
        # print(edit_inner)

        # print(
        #     "data",
        # )
        edited_model, model_info = self.alg.edit(edit_inner, cond, return_factors=True)
        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()

        # Edit!
        d = factors
        torch_factors = {k: torch.tensor(v) for k, v in d.items()}
        eli = 0
        edit_lrs = torch_factors["edit_lrs"]
        print(f"Base model weight checksum: {sum(p.sum() for p in model.parameters())}")
        with torch.no_grad():
            for n, p in model.named_parameters():
                uname, vname = f"{n}.u", f"{n}.v"
                if uname in torch_factors:
                    if return_orig_weights and n not in weights_copy:
                        weights_copy[n] = p.detach().clone()
                    with torch.no_grad():
                        p.copy_(edited_model.model.state_dict()[n])

        return model, weights_copy


class MendMultimodalRewriteExecutor(MendRewriteExecutor):
    def __init__(self):
        super().__init__()

    def init_model(self, model, tok, params: MENDMultimodalHparams):
        assert params.archive is not None or print(f"Training weights Needed....")

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        # add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        self.alg = MEND(self.model, params, lambda: deepcopy(self.model))
        d = torch.load(params.archive)

        self.alg.load_state_dict({k.replace("gtn.", "mend."): v for k, v in d["model"].items()})
        self.alg.to(torch.device(f"cuda:{params.device}"))

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in params.inner_params:
                p.requires_grad = False
        self.is_init = True

    def apply_to_model(
        self,
        model,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MENDMultimodalHparams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs,
    ):
        """
        Given a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns a dictionary of numpy arrays that specifies
        how mend will change the weights of the model.
        """

        if not self.is_init:
            self.init_model(model, tok, hparams)

        weights_copy = {}
        model = deepcopy(self.model) if copy else self.model

        # Define i/o
        src = [request["prompt"] for request in requests]
        trg = [(" " if request["target"][0] != " " else "") + request["target"] for request in requests]
        image = [request["image"] for request in requests]
        image = torch.stack(image, dim=0).to(model.device)
        text_input = [s + t for s, t in zip(src, trg)]

        if hparams.model_name == "minigpt4":
            prompts_len = [len(tok.encode(s, add_special_tokens=False)) for s in src]
            labels = tok(
                trg,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(model.device)
        else:
            prompts_len = [len(tok.encode(s)) for s in src]
            labels = tok(
                trg,
                return_tensors="pt",
            )["input_ids"].to(model.device)

        # Run MEND
        edit_inner = dict(image=image, text_input=text_input, labels=labels, prompts_len=prompts_len)
        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}

        self.alg.eval()
        edited_model, model_info = self.alg.edit(edit_inner, return_factors=True)
        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()

        # Edit!
        d = factors
        torch_factors = {k: torch.tensor(v) for k, v in d.items()}

        with torch.no_grad():
            for n, p in model.named_parameters():
                uname, vname = f"{n}.u", f"{n}.v"
                if uname in torch_factors:
                    if return_orig_weights and n not in weights_copy:
                        weights_copy[n] = p.detach().clone()
                    with torch.no_grad():
                        p.copy_(edited_model.model.state_dict()[n])

        if not keep_original_weight:
            weights_copy = {}

        return model, weights_copy


class MendPerRewriteExecutor(MendRewriteExecutor):
    def __init__(self):
        super().__init__()

    def apply_to_model(
        self,
        request,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        device,
        hparams: MENDHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs,
    ):
        if not self.is_init:
            self.init_model(model, tok, hparams)

        weights_copy = {}
        model = deepcopy(self.model) if copy else self.model

        self.alg.eval()
        edited_model, model_info = self.alg.edit(request["cond"], personality=True, return_factors=True)

        return edited_model, weights_copy
