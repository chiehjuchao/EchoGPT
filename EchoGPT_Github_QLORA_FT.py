# Use single GPU for QLoRA fine-tuning
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ## Dataset
import json

# Specify the filename
filename = "file_path1"

# Read and parse the JSON content
with open(filename, "r") as json_file:
    data = json.load(json_file)

# Specify the filename
filename = "file_path2"

# Read and parse the JSON content
with open(filename, "r") as json_file:
    val = json.load(json_file)

# Template
prompt_template = """ ### SYSTEM: You are a knowledgeable cardiologist. For the following echocardiography report findings specified by ### Input, please write a concise clinical summary with a minimal amount of text. Return your response as specified by ### CARDIOLOGIST.

### Input:
{input}

### CARDIOLOGIST:
{output}

"""

#Process training and validation datasets
processed_data = []
for j in data:
    processed_prompt = prompt_template.format(input=j["input"], output=j["output"])
    processed_data.append({"input": processed_prompt})

processed_val = []
for j in val:
    processed_prompt_val = prompt_template.format(input=j["input"], output=j["output"])
    processed_val.append({"input": processed_prompt_val})


import datasets
dataset = datasets.Dataset.from_list(processed_data)
val_data= datasets.Dataset.from_list(processed_val)


## Loading the model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map = "auto"
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# PEFT/LoRA configuration
from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)


## Loading the trainer
from transformers import TrainingArguments

output_dir = "output_dir_path"
per_device_train_batch_size = 1
gradient_accumulation_steps = 24
optim = "paged_adamw_32bit"
save_steps = 3979
logging_steps = 10
initial_learning_rate = 1e-3  # Starting learning rate
warmup_steps = 100            # Number of warmup steps
max_steps = -1             # Total number of training steps decided by num_train_epochs
num_train_epochs = 5
max_grad_norm = 0.3

# Use the linear scheduler for cosine/linear decay
lr_scheduler_type = "cosine"#"linear"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=initial_learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_steps=warmup_steps,  # Specify the warmup_steps directly
    lr_scheduler_type=lr_scheduler_type,  # Set the scheduler to 'cosine' or 'linear'
    # Additional arguments to enable mixed precision training and control evaluation and saving
    do_eval=True,#False,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    load_best_model_at_end=False,
)

#Pass everything to the trainer

from trl import SFTTrainer

max_seq_length = 700

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=val_data,
    peft_config=peft_config,
    dataset_text_field="input",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


#Also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


## Train the model
trainer.train()


