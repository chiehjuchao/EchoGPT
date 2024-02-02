"""##Load Data"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import json
# Specify the path to your JSON file
json_file_path = 'file_path'  # Update the file path as needed

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

"""##Load Model and Set up pipeline"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
print(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map = "cuda:0"
)

model.config.use_cache = False


# Load the tokenizer below
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Load the checkpoint and merge QLoRA weights
check_point_dir = "checkpoint_dir"
print(check_point_dir)

config = PeftConfig.from_pretrained(check_point_dir)

model = PeftModel.from_pretrained(model, check_point_dir)
merged_model = model.merge_and_unload()

import transformers
from langchain import HuggingFacePipeline

pipeline = transformers.pipeline(
    "text-generation", #task
    model=merged_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_new_tokens=300,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.1})

from langchain import PromptTemplate,  LLMChain

template = """
              SYSTEM: You are a knowledgeable cardiologist. For the following echocardiography report findings delimited by triple backquotes, please write a concise clinical summary with a minimal amount of text.   
              ```{text}```
              SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Initialize an empty list to store the responses
response_list = []
index = 0
from tqdm import tqdm

for processed_prompt in tqdm(data):
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    text = processed_prompt["input"]

    # Generate a response using llm_chain, and replace this line with your actual response generation logic
    response = llm_chain.run(text)

    # Create a dictionary to store the response and output
    response_dict = {
        "input":processed_prompt["input"],
        "output": processed_prompt["output"],
        "response": response
    }

    # Append the response dictionary to the list
    response_list.append(response_dict)

    # Increment the counter
    index += 1
    torch.cuda.empty_cache()

# Save the entire response_list to a single JSON file
import json
with open('file_path', 'w') as json_file:
    json.dump(response_list, json_file)