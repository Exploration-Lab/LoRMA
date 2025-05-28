import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional
# import evaluate

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`. "
                          "If False, will pad the samples dynamically when batching to the maximum length in the batch."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys:
                raise ValueError(f"Unknown task: {self.task_name}. Choose from {list(task_to_keys.keys())}.")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    apply_lorma: Optional[bool] = field(
        default=True,
        metadata={"help": "Apply LoRMA-plus to the model."}
    )
    lorma_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lorma_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "lorma_alpha"}
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    lorma_mode: Optional[str] = field(
        default="pre",
        metadata={"help": "`Mode to apply LoRMA-plus `pre`, `post`."}
    )
    do_rri: Optional[bool] = field(
        default=True,
        metadata={"help": "to apply RRI or not."}
    )
    lorma_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "dropout prob."}
    )
    adapted_weights: Optional[str] = field(
        default="qv",
        metadata={"help": "Adapted weights for LoRMA-plus."}
    )
    ##
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    def __post_init__(self):
        if self.apply_lorma == True:
            if self.lorma_mode not in ["pre", "post"]:
                raise ValueError(f"Unknown mode: {self.lorma_mode}. Choose from ['pre', 'post'].")



class LormaLinear_plus(nn.Module):

    def __init__(self, original_linear: nn.Linear, lorma_r: int, lorma_alpha: int, mode: str, do_rri: bool, lorma_dropout: float = 0.0):
        super().__init__()
        
        assert mode in ["pre", "post"], "mode must be either pre, post"
        self.mode = mode
        self.original_layer = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.lorma_r = lorma_r
        self.scaling = lorma_alpha / lorma_r
        
        self.do_rri = do_rri 
        if lorma_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lorma_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # initialize lora_A and lora_B params
        if self.mode == "pre":
            self.lora_B = nn.Parameter(original_linear.weight.new_zeros(self.out_features, self.lorma_r))
            self.lora_A = nn.Parameter(original_linear.weight.new_zeros(self.lorma_r, self.out_features))
        else:
            # mode is "post"
            self.lora_B = nn.Parameter(original_linear.weight.new_zeros(self.in_features, self.lorma_r))
            self.lora_A = nn.Parameter(original_linear.weight.new_zeros(self.lorma_r, self.in_features))
                    
        # initializing the lora params
        if self.do_rri:
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            with torch.no_grad():
                self.lora_A.copy_(torch.linalg.pinv(self.lora_B))
                self.lora_A.data /= math.sqrt(self.scaling)
                self.lora_B.data /= math.sqrt(self.scaling)

    def forward(self, x):
        # logger.info("Called LormaLinear_plus forward")
        if self.mode == "pre":
            if self.do_rri:
                adapted_weight = (self.lora_B @ (self.lora_A @ self.original_layer.weight)) * self.scaling + self.original_layer.weight
            else:
                adapted_weight = (self.lora_B @ (self.lora_A @ self.original_layer.weight)) * self.scaling
            output = F.linear(self.lora_dropout(x), adapted_weight, self.original_layer.bias)
        else:
            # mode is "post"
            if self.do_rri:
                adapted_weight = ((self.original_layer.weight @ self.lora_B) @ self.lora_A) * self.scaling + self.original_layer.weight
            else:
                adapted_weight = (self.original_layer.weight @ self.lora_B) @ self.lora_A
            output = F.linear(self.lora_dropout(x), adapted_weight, self.original_layer.bias)
        
        return output
    
            
def apply_lorma(model, model_args):
    lorma_r = model_args.lorma_r
    mode = model_args.lorma_mode
    do_rri = model_args.do_rri
    dropout = model_args.lorma_dropout
    alpha = model_args.lorma_alpha
    
    adapted_weights = model_args.adapted_weights.lower()  # Ensure case-insensitive
    
    logger.info(f"Applying LoRMA with r={lorma_r}, do_rri={do_rri}, mode={mode} on adapted_weights={adapted_weights}")
    
    if lorma_r <= 0:
        raise ValueError("lora_r must be greater than 0")
    
    # Map characters to attention layer patterns
    pattern_map = {
        "q": "attention.self.query",
        "k": "attention.self.key",
        "v": "attention.self.value",
        "o": "attention.output.dense",
    }
    
    # Determine target patterns from adapted_weights string
    target_patterns = []
    for c in adapted_weights:
        if c in pattern_map:
            target_patterns.append(pattern_map[c])
    
    if not target_patterns:
        raise ValueError(f"Invalid adapted_weights: {adapted_weights}. Must contain at least one of q/k/v/o")
    
    # Find and replace matching Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer matches ANY of the target patterns
            for pattern in target_patterns:
                if pattern in name:
                    parent_module = model
                    sub_names = name.split(".")
                    for sub_name in sub_names[:-1]:
                        parent_module = getattr(parent_module, sub_name)
                    child_name = sub_names[-1]
                    
                    original_linear = getattr(parent_module, child_name)
                    lorma_linear = LormaLinear_plus(
                        original_linear, lorma_r, alpha, mode, do_rri, dropout
                    )
                    
                    setattr(parent_module, child_name, lorma_linear)
                    logger.info(f"Applied LoRMA to {name}")
                    break  # Avoid duplicate processing
    
    return model

# This function is from the original LoRA implementation
def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError
    
    
def main():
    # clearing torch cache

   #  torch.cuda.empty_cache()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # INITAL JARGON
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # DATA LOADING
    # Load the dataset (ASSUMPTION: not from any file)
    datasets = load_dataset("glue", data_args.task_name)
    if data_args.task_name == "mnli":
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
            logger.info(f"Label list: {label_list}")
        else:
            num_labels = 1
    else:
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = datasets["train"].unique("label")
            label_list.sort()
            logger.info(f"Label list: {label_list}")
            num_labels = len(label_list)
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        num_labels=num_labels,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # cls_dropout=training_args.cls_dropout,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if model_args.apply_lorma and model_args.lorma_r > 0:
        model = apply_lorma(model, model_args)
        logger.info("Applied LoRMA to the model")

    mark_only_lora_as_trainable(model, "none")
    
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad = True
    logger.info("Marked classifier layer as trainable")

    
    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))


    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
        
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    training_args.label_names=["labels"]
    
    # renaming columns required in new version
    
    if training_args.do_train:
        train_dataset = train_dataset.rename_column("label", "labels")
    if training_args.do_eval:
        eval_dataset = eval_dataset.rename_column("label", "labels")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
            
    
    trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n)
    
    logger.info("*"*80)
    logger.info(f"trainable_params: {trainable_params}")
    logger.info("*"*80)

    
    training_args.report_to = ["tensorboard"]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.can_return_loss = True

    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()
