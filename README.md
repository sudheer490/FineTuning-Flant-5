# Fine-Tune FLAN-T5 with Reinforcement Learning (PPO) and PEFT to Generate Less-Toxic Summaries

In this project, we aim to fine-tune a FLAN-T5 model to generate less toxic content using Meta AI's hate speech reward model. The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. We utilize Proximal Policy Optimization (PPO) to fine-tune the model, reducing its toxicity.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Model Loading and Preparation](#model-loading)
    - Load Data and FLAN-T5 Model
    - Prepare Reward Model
    - Evaluate Toxicity
4. [Fine-Tuning with PPO](#fine-tuning)
    - Initialize PPOTrainer
    - Fine-Tune the Model
    - Quantitative Evaluation
    - Qualitative Evaluation
5. [Results and Analysis](#results)
6. [Conclusion](#conclusion)

<a name="introduction"></a>
## 1. Introduction

This project aims to fine-tune a pre-trained FLAN-T5 model to generate less toxic summaries. We use Meta AI's hate speech reward model as the evaluator and Proximal Policy Optimization (PPO) for reinforcement learning. This README provides an overview of the process, including setting up dependencies, model loading, preparation, and the fine-tuning process.

<a name="setup"></a>
## 2. Setup

First, ensure that the correct kernel and environment are set up.

The expected instance type for running this project is `ml-m5-2xlarge`. Verify the instance type:

```python
import os
instance_type_expected = 'ml-m5-2xlarge'
instance_type_current = os.environ.get('HOSTNAME')

print(f'Expected instance type: instance-datascience-{instance_type_expected}')
print(f'Currently chosen instance type: {instance_type_current}')
assert instance_type_expected in instance_type_current, f'ERROR: Please select the correct instance type.'
```

Install the required packages:

```bash
%pip install -U datasets==2.17.0
%pip install --upgrade pip
%pip install torch==1.13.1 torchdata==0.5.1 transformers==4.27.2 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0
%pip install git+https://github.com/lvwerra/trl.git@25fa1bd
```

<a name="model-loading"></a>
## 3. Model Loading and Preparation

### 3.1 Load Data and FLAN-T5 Model

We work with the Hugging Face dataset `DialogSum` and the pre-trained model FLAN-T5.

```python
from datasets import load_dataset
model_name="google/flan-t5-base"
huggingface_dataset_name = "knkarthick/dialogsum"

dataset_original = load_dataset(huggingface_dataset_name)
```

### 3.2 Prepare Reward Model

We use Meta AI's RoBERTa-based hate speech model as the reward model. The model outputs logits to predict the probabilities of two classes: "nothate" and "hate".

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)
```

### 3.3 Evaluate Toxicity

The toxicity evaluator computes a toxicity score, ranging from 0 to 1.

```python
import evaluate
toxicity_evaluator = evaluate.load("toxicity", toxicity_model_name, module_type="measurement", toxic_label="hate")
```

<a name="fine-tuning"></a>
## 4. Fine-Tuning with PPO

### 4.1 Initialize PPOTrainer

To initialize PPOTrainer, we need a data collator:

```python
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
```

### 4.2 Fine-Tune the Model

The fine-tuning loop consists of the following steps:

1. Generate responses from the policy LLM (PEFT model).
2. Obtain sentiment scores for the query-response pairs using the hate speech RoBERTa model.
3. Optimize the policy with PPO using the (query, response, reward) triplet.

```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    ppo_epochs=1,
    mini_batch_size=4,
    batch_size=16
)
ppo_trainer = PPOTrainer(config=config, model=ppo_model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset["train"], data_collator=collator)
```

### 4.3 Quantitative Evaluation

We evaluate the toxicity of the RL fine-tuned model using the toxicity evaluator:

```python
mean_after_detoxification, std_after_detoxification = evaluate_toxicity(model=ppo_model, toxicity_evaluator=toxicity_evaluator, tokenizer=tokenizer, dataset=dataset["test"], num_samples=10)
print(f'toxicity [mean, std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')
```

### 4.4 Qualitative Evaluation

We compare the generated responses before and after detoxification using the frozen reference model and the PPO fine-tuned model:

```python
import pandas as pd
from tqdm import tqdm

df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(drop=True)
df_compare_results_sorted
```

<a name="results"></a>
## 5. Results and Analysis

The quantitative and qualitative evaluation metrics show significant improvement in reducing toxicity after fine-tuning with PPO. The `reward_diff` metric indicates that the detoxified model generates less harmful content compared to the reference model.

<a name="conclusion"></a>
## 6. Conclusion

This project demonstrates the effectiveness of using reinforcement learning techniques like PPO along with a hate speech reward model to fine-tune a FLAN-T5 model for generating less-toxic content. The reduced toxicity in generated summaries is verified both quantitatively and qualitatively. Future work can explore different reward models and larger datasets for further improvements.
