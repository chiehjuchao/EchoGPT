---
library_name: peft
base_model: meta-llama/Llama-2-7b-chat-hf
---

# Model Card for Model ID

EchoGPT is a fine-tuned LLM that handles echocardiography report summarization tasks, generating clinically relevant "Final Impressions" from "Findings."

## Model Details

### Model Description

- Developed by: Chieh-Ju Chao
- Language(s) (NLP): English

- Repository: https://github.com/chiehjuchao/EchoGPT
- Paper: https://www.medrxiv.org/content/10.1101/2024.01.18.24301503v3

## How to Get Started with the Model

Use the code below to get started with the model.

Clone repo:  

```bash
git clone https://github.com/chiehjuchao/EchoGPT.git

cd EchoGPT
```
Set environment:
```bash
pip install -r requirements.txt
```

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

## Citation

Chao, C.-J., Banerjee, I., Arsanjani, R., Ayoub, C., Tseng, A., Delbrouck, J.-B., Kane, G. C., Lopez-Jimenez, F., Attia, Z., Oh, J. K., Erickson, B., Fei-Fei, L., Adeli, E. & Langlotz, C. (2024). Evaluating Large Language Models in Echocardiography Reporting: Opportunities and Challenges. MedRxiv, 2024.01.18.24301503. https://doi.org/10.1101/2024.01.18.24301503


