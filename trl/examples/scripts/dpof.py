# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/dpof.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpof.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.trainer import DPOfTrainer as DPOTrainer

import wandb


@dataclass
class ScriptArguments:
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    max_length: int = field(default=1024, metadata={"help": "max length of each sample"})
    max_prompt_length: int = field(default=128, metadata={"help": "max length of each sample's prompt"})#128
    max_target_length: int = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )#128
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    generate_during_eval: bool = field(default=False, metadata={"help": "Generate during evaluation"})
    dual_lr: float = field(default=1.0, metadata={"help": "Dua; Learning Rate"})

    resilient_alpha: float = field(default=2.0, metadata={"help": "Dual Weight Decay Coefficient"})
    loss_tolerance: float = field(default=1e-3, metadata={"help": "Loss Tolerance"})
    algorithm: str = field(default="erm", metadata={"help": "Algorithm can be 'erm', 'clamped' or 'feasible'"})
    dataset: str = field(default="hh", metadata={"help": "The dataset to load. Can be 'hh' or 'ultra'."})
    train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)

def get_ultra(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load HuggingFaceH4/ultrafeedback_binarized from Hugging Face. 
    """
   
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split+"_prefs", cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"][1]["content"],
            "rejected": sample["rejected"][1]["content"],
        }
    dataset = dataset.remove_columns('messages')
    return dataset.map(split_prompt_and_responses)
#
    #return dataset

def chatml_format(example):
    """
    From mlabnonnes example nbook: https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing
    """
    # Format system
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": example['input']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def get_orca(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None, distilled: bool = True, val_fraction: float = 0.2) -> Dataset:
    if not distilled:
        raise NotImplementedError
    else:
        dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train", cache_dir=cache_dir)
        dataset = dataset.filter(
            lambda r: 
                r["status"] != "tie" and 
                r["chosen_score"] >= 8 and 
                not r["in_gsm8k_train"]
        )
        #dataset = dataset.remove_columns(['system', 'generations', 'order', 'labelling_model', 'labelling_prompt', 'raw_labelling_response','rating', 'rationale', 'status', 'original_chosen', 'original_rejected', 'chosen_score', 'in_gsm8k_train'])
        # Save columns
        original_columns = dataset.column_names
        
        #dataset  = dataset.shuffle()
        if split=="test":
            dataset = dataset.select(range(int(len(dataset)*val_fraction)))
        if split=="train":
            dataset = dataset.select(range(int(len(dataset)*val_fraction), len(dataset)))

        dataset = dataset.map(chatml_format, remove_columns=original_columns)
        if sanity_check:
            dataset = dataset.select(range(min(len(dataset), 16)))
        #dataset = dataset.rename_columns({"question": "input"})
        #dataset = dataset.remove_columns(['system', 'generations', 'order', 'labelling_model', 'labelling_prompt', 'raw_labelling_response','rating', 'rationale', 'status', 'original_chosen', 'original_rejected', 'chosen_score', 'in_gsm8k_train'])
        return dataset
        

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.num_train_epochs = args.train_epochs
    if hasattr(training_args, "gradient_checkpointing_kwargs"):
        training_args.gradient_checkpointing_kwargs ={"use_reentrant":  False}
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    if hasattr(quantization_config, "bnb_4bit_compute_dtype"):
        quantization_config.bnb_4bit_compute_dtype=torch.bfloat16
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    #model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    if args.dataset == "orca":
        # from mlabonnes example nbook https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    ################
    # Dataset
    ################
    if args.dataset == "hh":
        train_dataset = get_hh("train", sanity_check=args.sanity_check)
        eval_dataset = get_hh("test", sanity_check=args.sanity_check)
    elif args.dataset == "ultra":
        train_dataset = get_ultra("train", sanity_check=args.sanity_check)
        eval_dataset = get_ultra("test", sanity_check=args.sanity_check)
    elif args.dataset == "orca":
        train_dataset = get_orca("train", sanity_check=args.sanity_check)
        eval_dataset = get_orca("test", sanity_check=args.sanity_check)
    else:
        raise NotImplementedError

    wandb.init(project="feasible-rlhf", config={**args.__dict__, **training_args.__dict__, **model_config.__dict__})

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        max_prompt_length=args.max_prompt_length,
        generate_during_eval=args.generate_during_eval,
        peft_config=get_peft_config(model_config),
        dual_lr=args.dual_lr,
        resilient_alpha=args.resilient_alpha,
        loss_tolerance=args.loss_tolerance,
        algorithm=args.algorithm,
    )
    trainer.train()   
    #trainer.save_model(training_args.output_dir)
    trainer.train_dataset = trainer.train_dataset.remove_columns('indexes')
    results = trainer.evaluation_loop(
        trainer.get_train_dataloader(),
        "eval/train",
        metric_key_prefix = "eval/train",
    )
    #wandb.log(results[2])

    results = trainer.evaluation_loop(
        trainer.get_eval_dataloader(),
        "eval/val",
        metric_key_prefix = "eval/val",
    )
    #wandb.log(results[2])

    wandb.log({"multipliers/hist": wandb.Histogram(trainer.multipliers.cpu().detach().numpy())})
