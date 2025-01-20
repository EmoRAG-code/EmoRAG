# EmoRAG Defense Code 🛡️

## Overview 📜

This project contains the implementation of defense mechanisms against the **EmoRAG** attack, as detailed in the paper titled "**EmoRAG: Demystifying Emoticons Poisoning in Retrieval-Augmented Large Language Models**". 😈

The **EmoRAG** attack exploits seemingly innocent emoticons (like `(@_@)`) to subtly manipulate the retrieval process in **Retrieval-Augmented Generation (RAG)** systems. The attack can compromise the accuracy and fairness of language model-generated outputs by causing them to retrieve malicious or biased information. This repository offers effective defenses to mitigate this risk. 💪

⚠️ **Please note**: In order to adhere to ethical guidelines, we have not open-sourced our attack code. All the details of the attack methodology are available in the paper. However, we have open-sourced the entire defense code, ensuring that the research community can advance the security of RAG systems.

![EmoRAG Diagram](image/EmoRAG.png) 

## EmoRAG Attack 🔓

The **EmoRAG** attack works by injecting emoticons into queries that are processed by **RAG systems**. This manipulation causes the system to retrieve irrelevant, malicious, or biased texts that include the same emoticons. A single emoticon can significantly alter the query’s semantic embeddings, making the retriever prioritize harmful content. 🚨

For more detailed explanations on how the attack works, please refer to the original paper. 📝

## Defenses Implemented 🛡️

This repository implements the following defense mechanisms against the EmoRAG attack:

1. **Emoticon Detection**: A model trained to detect malicious texts that contain suspicious emoticons used in backdoor attacks. 🔍  
   You can access the pre-trained model for emoticon detection on [Hugging Face](https://huggingface.co/qwedxzawsedr/EmoRAG_detect) 🤖.  
   Additionally, the dataset used for training this model is available on [Hugging Face Datasets](https://huggingface.co/datasets/qwedxzawsedr/emorag_defense) 📊.
2. **Query Disinfection**: Sanitizes queries before they enter the RAG system to remove harmful emoticons. 🚿
3. **Embedding Regularization**: Adjusts the query embeddings to reduce the impact of emoticon perturbations on the retrieval and generation processes. 🔧

## How to Use 🚀

### 1. Clone the Repository
Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/EmoRAG-Defense.git
cd EmoRAG-Defense
```
### 2. Install Dependencies 🧑‍💻

Make sure all the necessary libraries are installed:

```bash
pip install -r requirements.txt
```
### 3. Train the Defense Model 🏋️‍♂️
To train the model for detecting emoticon-based backdoor attacks, use the following command:

```bash
python training_model.py
```

