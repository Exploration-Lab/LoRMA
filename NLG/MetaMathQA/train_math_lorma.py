#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Zheng Yuan and Hongyi Yuan

#    Adapting the above code-base.

import os
import copy

import sys

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import random;random.seed(42)
import torch.nn as nn
import torch.nn.functional as F

# from peft import get_peft_model, LoraConfig, VeraConfig, BOFTConfig
from tqdm import tqdm
from functools import partial, reduce

import sys


def get_target_modules_list(model, target_modules):
    target_names = []
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules:
            target_names.append(name)
    return target_names


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
#### 28
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class LoRMAArguments:
    lorma_r: int = field(default=8, metadata={"help": "rannk of lorma adapters"})
    lorma_alpha: int = field(default=16, metadata={"help": "alpha of lorma adapters"})
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lorma_mode: str = field(default="pre", metadata={"help": "lorma mode, choose from 'pre', 'post'"})
    rank_inflation: str = field(default="rri", metadata={"help": "rank inflation, choose from 'rri', 'shift', 'none'"})
    lorma_dropout: float = field(default=0.0, metadata={"help": "dropout of lorma adapters"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        try:
            data_path = data_path_map[data_path]
        except:
            data_path = data_path
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        list_data_dict = random.sample(list_data_dict,  len(list_data_dict))
        list_data_dict = list_data_dict[:data_args.data_length]

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'].split('\n')[0], 'input':get_input(data['query']), 'output':data['response']} for data in list_data_dict]

        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class MultLoRALinear(torch.nn.Module):
    def __init__(self, original_linear, lorma_r, lorma_alpha, lorma_mode, rank_inflation, lorma_dropout):
        super().__init__()
        
        self.original_linear = original_linear
        self.lorma_r = lorma_r
        self.lorma_alpha = lorma_alpha
        self.scaling = lorma_alpha / lorma_r
        self.lorma_mode = lorma_mode
        self.rank_inflation = rank_inflation
        if lorma_dropout > 0:
            self.dropout = torch.nn.Dropout(lorma_dropout)
        else:
            self.dropout = lambda x: x
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        if self.lorma_mode == "pre":
            self.lorma_B = nn.Parameter(original_linear.weight.new_zeros((self.out_features, self.lorma_r)))
            self.lorma_A = nn.Parameter(original_linear.weight.new_zeros((self.lorma_r, self.out_features)))
        elif self.lorma_mode == "post":
            self.lorma_B = nn.Parameter(original_linear.weight.new_zeros((self.in_features, self.lorma_r)))
            self.lorma_A = nn.Parameter(original_linear.weight.new_zeros((self.lorma_r, self.in_features)))
        else:
            raise ValueError(f"lorma_mode {self.lorma_mode} not supported")
        
        if self.rank_inflation == "rri":
            nn.init.zeros_(self.lorma_B)
            nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))
        elif self.rank_inflation == "shift":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lorma_B, a=math.sqrt(5))
                self.lorma_B[:, 0] = 1.0
                self.lorma_A.data.zero_()
                self.lorma_A.data[0,0] = 1.0
                self.lorma_A.data /= math.sqrt(self.scaling)
                self.lorma_B.data /= math.sqrt(self.scaling)
                
                # also shifting index in the case rank_inflation == "shift"
                self.shifting_matrices = {}
                
        elif self.rank_inflation == "none":
            nn.init.kaiming_uniform_(self.lorma_B, a=math.sqrt(5))
            with torch.no_grad():
                self.lorma_A.copy_(torch.linalg.pinv(self.lorma_B))
                self.lorma_A.data /= math.sqrt(self.scaling)
                self.lorma_B.data /= math.sqrt(self.scaling)
        else:
            raise ValueError(f"rank_inflation {self.rank_inflation} not supported")
        
    def shift(self, M):
        shape = M.shape
        device = M.device
        key = shape, device
        if key not in self.shifting_matrices:
            n_rows, n_cols = shape
            row_indices = torch.arange(n_rows, device=device).view(-1, 1)
            col_indices = torch.arange(n_cols, device=device).view(1, -1)
            # Compute shift indices using broadcasting and modulo operation
            indices = (col_indices - row_indices) % n_cols
            self.shifting_matrices[key] = indices
        return torch.gather(M, 1, self.shifting_matrices[key])
    
    def forward(self, x):
        if self.lorma_mode == "pre":
            if self.rank_inflation == "rri":
                adapted_weights = (self.lorma_B @ (self.lorma_A @ self.original_linear.weight)) * self.scaling + self.original_linear.weight
            elif self.rank_inflation == "shift":
                adapted_weights = self.shift(self.lorma_B @ self.lorma_A) @ self.original_linear.weight * self.scaling
            else:
                adapted_weights = (self.lorma_B @ (self.lorma_A @ self.original_linear.weight)) * self.scaling
                
            output = F.linear(self.dropout(x), adapted_weights, self.original_linear.bias)
            
            return output
        elif self.lorma_mode == "post":
            if self.rank_inflation == "rri":
                adapted_weights = ((self.original_linear.weight @ self.lorma_B) @ self.lorma_A) * self.scaling + self.original_linear.weight
            elif self.rank_inflation == "shift":
                adapted_weights = self.original_linear.weight @ self.shift(self.lorma_B @ self.lorma_A) * self.scaling
            else:
                adapted_weights = ((self.original_linear.weight @ self.lorma_B) @ self.lorma_A) * self.scaling
                
            output = F.linear(self.dropout(x), adapted_weights, self.original_linear.bias)
            
            return output
        else:
            raise ValueError(f"lorma_mode {self.lorma_mode} not supported")
        
def apply_lorma(model, lorma_args):
    lorma_r = lorma_args.lorma_r
    lorma_alpha = lorma_args.lorma_alpha
    rank_inflation = lorma_args.rank_inflation
    dropout = lorma_args.lorma_dropout
    lorma_mode = lorma_args.lorma_mode
    
    print(f"Applying LoRMA with rank {lorma_r}, alpha {lorma_alpha}, mode {lorma_mode}, rank_inflation {rank_inflation}, dropout {dropout}")
    
    target_modules_list = get_target_modules_list(model, lorma_args.target_modules)
    
    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        
        lorma_target = MultLoRALinear(
            original_linear=target,
            lorma_r=lorma_r,
            lorma_alpha=lorma_alpha,
            lorma_mode=lorma_mode,
            rank_inflation=rank_inflation,
            lorma_dropout=dropout
        )
        
        parent.__setattr__(target_name, lorma_target)
        print(f"Replaced {target_path} with LoRMA layer")
        
    return model

def mark_only_lorma_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lorma" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    else:
        raise NotImplementedError
    
   
def merge_lorma_and_unload(model, lorma_args):
    print("Replacing LoRMA layers with new Linear layers")
    
    target_modules_list = get_target_modules_list(model, lorma_args.target_modules)
    
    print("target_modules_list", target_modules_list)
    
    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        
        in_dim = target.in_features
        out_dim = target.out_features
        
        if target.original_linear.bias is None:
            lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        else:
            lin = torch.nn.Linear(in_dim, out_dim, bias=True)
            lin.bias.data = target.original_linear.bias.data
        
        if lorma_args.lorma_mode == "pre":
            if lorma_args.rank_inflation == "rri":
                lin.weight.data = (target.lorma_B @ (target.lorma_A @ target.original_linear.weight)) * target.scaling + target.original_linear.weight
            elif lorma_args.rank_inflation == "shift":
                lin.weight.data = target.shift(target.lorma_B @ target.lorma_A) @ target.original_linear.weight * target.scaling
            else:
                lin.weight.data = (target.lorma_B @ (target.lorma_A @ target.original_linear.weight)) * target.scaling
        elif lorma_args.lorma_mode == "post":
            if lorma_args.rank_inflation == "rri":
                lin.weight.data = ((target.original_linear.weight @ target.lorma_B) @ target.lorma_A) * target.scaling + target.original_linear.weight
            elif lorma_args.rank_inflation == "shift":
                lin.weight.data = target.original_linear.weight @ target.shift(target.lorma_B @ target.lorma_A) * target.scaling
            else:
                lin.weight.data = ((target.original_linear.weight @ target.lorma_B) @ target.lorma_A) * target.scaling
        
        parent.__setattr__(target_name, lin)
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoRMAArguments))
    model_args, data_args, training_args, lorma_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_args.data_length = int(remaining_args[1])

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    ).to("cuda")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama-2" in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # applying lorma
    model = apply_lorma(model, lorma_args)
    mark_only_lorma_as_trainable(model, bias="none")

    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    print(f"Training args: {training_args}")
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()

    trainer.save_state()
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    
    merge_lorma_and_unload(model, lorma_args)

    for param in model.parameters():
        param.data = param.data.contiguous()
        
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()