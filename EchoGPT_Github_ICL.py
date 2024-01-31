import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
# Specify the path to your JSON file
json_file_path = 'file_path'  # Update the file path as needed

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

for entry in data:
    # Temporary list to store formatted examples
    formatted_examples = []

    for example in entry["examples"]: #Contains m ICL examples (m= 1,2,4,8,...)
        # Format input and output strings
        formatted_input = f"###FINDINGS: {example['input']}"
        formatted_output = f"###SUMMARY: {example['output']}"

        # Combine the formatted input and output
        combined_example = f"{formatted_input}\n{formatted_output}"

        # Add the combined example to the temporary list
        formatted_examples.append(combined_example)

    # Replace the original examples with the formatted versions
    entry['examples'] = "\n\n".join(formatted_examples)

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

import transformers
from langchain import HuggingFacePipeline

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
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
              Use the following examples delimited by triple backquotes to guide word choice.
              ```{examples}```
              SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text", "examples"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Initialize an empty list to store the responses
response_list = []
index = 0
from tqdm import tqdm

import datasets
from langchain import PromptTemplate, LLMChain

# Convert processed_data to a Hugging Face dataset
dataset = datasets.Dataset.from_list(data)

# Function to process each prompt
def process_prompt(processed_prompt):
    # Extract 'text' and 'examples' from the processed prompt
    text = processed_prompt["test_data"]["input"]
    examples = processed_prompt.get("examples", "")  # Use an empty string if no examples are provided

    # Update the PromptTemplate to include 'examples' as an input variable
    prompt = PromptTemplate(template=template, input_variables=["text", "examples"])

    # Create the LLMChain with the updated prompt
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Generate a response using llm_chain with both text and examples
    response = llm_chain.run(text=text, examples=examples)
    
    # Clear cache
    torch.cuda.empty_cache()

    # Return a dictionary with the response and output
    return {
        "input": text,
        "response": response,
        "output": processed_prompt["test_data"]["output"]
    }

# Apply the function to each element of the dataset in parallel
response_dataset = dataset.map(process_prompt)

response_list = [{"input": item["input"], "output": item["output"], "response": item["response"]} for item in response_dataset]

# Save the entire response_list to a single JSON file
import json
with open('file_path', 'w') as json_file:
    json.dump(response_list, json_file)


