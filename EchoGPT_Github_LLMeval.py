import json
# Specify the path to your JSON file
json_file_path = 'file_path'  # Update the file path as needed

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

#Pre-processing: Remove repetitive contents after '\n\n' 
for item in data:
    response = item.get("response", "")
    # Find the position of '\n\n'
    cut_off_index = response.find('\n\n')
    if cut_off_index != -1:
        # Keep only the part of the string before '\n\n'
        item["response"] = response[:cut_off_index]
    
#Import modules
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import bert_score
from radgraph import RadGraph, F1RadGraph  # Import RadGraph and F1RadGraph
import pandas as pd  # Import pandas for creating DataFrames

# Download the necessary NLTK resources
nltk.download('punkt')

# Define your calculate_scores function
def calculate_scores(response, output):
    # Tokenize the response and output using NLTK
    response_tokens = nltk.word_tokenize(response)
    output_tokens = nltk.word_tokenize(output)

    # Calculate BLEU score
    bleu_score = sentence_bleu([response_tokens], output_tokens)

    # Calculate METEOR Score
    meteor_score_value = meteor_score([response_tokens], output_tokens)

    # Calculate ROUGE-L score
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(response, output)
    rouge_l_score = rouge_scores['rougeL'].fmeasure

    # Calculate BERT score
    _, _, bert_score_value = bert_score.score([output], [response], lang="en")

    return bleu_score, meteor_score_value, rouge_l_score, bert_score_value

# Lists to store individual scores
bleu_scores = []
meteor_scores = []
rouge_l_scores = []
bert_scores = []  
f1radgraph_scores = []

#Calculate RadGraph F1 score
f1radgraph = F1RadGraph(reward_level="all").to(device)

for item in tqdm(data):
    response = item.get("response", [])
    output = item.get("output", [])
        
    try:
        # Calculate the value for each item
        value = f1radgraph([response], [output])[0][2]
    except Exception as e:
        print(f"Error calculating value: {e}")
        value = np.nan
        
    # Append the value to the list
    f1radgraph_scores.append(value)

mean_f1radgraph = 100*np.mean(f1radgraph_scores)
std_dev_f1radgraph = 100*np.std(f1radgraph_scores)
print(f"Mean F1RadGraph Score: {mean_f1radgraph:.1f} ± {std_dev_f1radgraph:.1f}")

# Loop through the dataset and calculate scores for each pair
for data_entry in tqdm(data):
    response = data_entry['response']
    output = data_entry['output']

    bleu, meteor, rouge_l, bert_score_value = calculate_scores(response, output)

    bleu_scores.append(bleu)
    meteor_scores.append(meteor)
    rouge_l_scores.append(rouge_l)
    bert_scores.append(bert_score_value)

# Create a DataFrame to store the raw scores
bert_scores = [bert_score_value.numpy()[0] for bert_score_value in bert_scores]

raw_scores_data = {
    'BLEU Score': bleu_scores,
    'METEOR Score': meteor_scores,
    'ROUGE-L Score': rouge_l_scores,
    'BERT Score': bert_scores,
    'F1RadGraph Score': f1radgraph_scores,
}

raw_scores_df = pd.DataFrame(raw_scores_data)

# Save the raw scores DataFrame to an Excel file
raw_scores_df.to_excel('file_path', index=False)

# Calculate the mean and standard deviation of each score
mean_bleu = 100*np.mean(bleu_scores)
std_dev_bleu = 100*np.std(bleu_scores)

mean_meteor = 100*np.mean(meteor_scores)
std_dev_meteor = 100*np.std(meteor_scores)

mean_rouge_l = 100*np.mean(rouge_l_scores)
std_dev_rouge_l = 100*np.std(rouge_l_scores)

mean_bert = 100*np.mean(bert_scores)
std_dev_bert = 100*np.std(bert_scores)

mean_f1radgraph = 100*np.mean(f1radgraph_scores)
std_dev_f1radgraph = 100*np.std(f1radgraph_scores)

# Print mean and standard deviation with two decimal places
print(f"Mean BLEU Score: {mean_bleu:.1f} ± {std_dev_bleu:.1f}")
print(f"Mean METEOR Score: {mean_meteor:.1f} ± {std_dev_meteor:.1f}")
print(f"Mean ROUGE-L Score: {mean_rouge_l:.1f} ± {std_dev_rouge_l:.1f}")
print(f"Mean BERT Score: {mean_bert:.1f} ± {std_dev_bert:.1f}")
print(f"Mean F1RadGraph Score: {mean_f1radgraph:.1f} ± {std_dev_f1radgraph:.1f}")