

---

# Named Entity Recognition (NER) with Hugging Face Transformers

## Overview

This notebook demonstrates the implementation of Named Entity Recognition (NER) using the Hugging Face `transformers` library, specifically for a custom dataset. The goal of this project is to classify tokens (words) in text into predefined categories such as Person, Location, Organization, etc. The model is fine-tuned using a pre-trained transformer model to predict these categories on the input text.

## Objectives

- Fine-tune a transformer-based model for NER.
- Align token labels with subword tokenization.
- Evaluate the performance of the model using common metrics (Precision, Recall, F1-Score, and Accuracy).

## Dataset

The dataset used for training consists of sentences where each token (word) is labeled with a Named Entity Recognition (NER) tag. The dataset can be customized by replacing the input tokens and their corresponding labels in the notebook. The labels correspond to different named entity types such as `PER` (person), `LOC` (location), and `ORG` (organization), or other custom labels depending on your use case.

## Libraries Used

- **transformers**: Hugging Face library for working with state-of-the-art NLP models.
- **datasets**: Hugging Face library for loading and processing datasets.
- **torch**: PyTorch for model training and evaluation.
- **sklearn**: For evaluating model metrics like precision, recall, and F1-score.

## Model

The model used in this notebook is a pre-trained transformer from the Hugging Face Model Hub (such as `bert-base-cased` or `roberta-base`) fine-tuned for the NER task. The model is fine-tuned using the tokenized input text and corresponding NER labels.

## Methodology

1. **Data Preprocessing**: 
    - The dataset is tokenized using the Hugging Face tokenizer.
    - The tokenized labels are aligned with the tokens, considering the subword tokenization process.
    - Labels are padded to match the sequence length after tokenization.

2. **Fine-Tuning**: 
    - The pre-trained transformer model is fine-tuned on the labeled data.
    - We use standard training procedures such as using the Adam optimizer, learning rate scheduling, and checkpointing.

3. **Evaluation**:
    - The model is evaluated using Precision, Recall, F1-Score, and Accuracy metrics.
    - Results are reported after every epoch, showcasing the performance improvement over time.

4. **Metrics**:
    - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
    - **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
    - **F1-Score**: The weighted average of Precision and Recall.
    - **Accuracy**: The overall accuracy of the model predictions.

## Code Explanation

### 1. Tokenization and Label Alignment

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
```

In this function, the input text is tokenized using a Hugging Face tokenizer. The labels are aligned with the tokens, especially in the case of subword tokenization where a word might be split into multiple tokens. The `align_labels_with_tokens` function is responsible for ensuring that each token receives the correct label.

### 2. Model Training

```python
trainer.train()
```

Here, the model is fine-tuned using the training data. The `trainer` is a Hugging Face utility that handles training loops and evaluation for models. The model is updated after every batch, improving its ability to classify tokens into correct named entities.

### 3. Model Evaluation

The model is evaluated on a validation set, and after each epoch, metrics like Precision, Recall, F1-Score, and Accuracy are printed:

```python
epoch 0: {'precision': 0.944, 'recall': 0.936, 'f1': 0.940, 'accuracy': 0.986}
epoch 1: {'precision': 0.944, 'recall': 0.936, 'f1': 0.940, 'accuracy': 0.986}
epoch 2: {'precision': 0.944, 'recall': 0.936, 'f1': 0.940, 'accuracy': 0.986}
```

The evaluation results show the model’s performance over three epochs, with stable precision, recall, F1 score, and accuracy, indicating the model's consistency in learning.

## Results

The model achieves the following evaluation results for all epochs:

- **Precision**: 0.944
- **Recall**: 0.936
- **F1-Score**: 0.940
- **Accuracy**: 0.986

These results demonstrate the high performance of the fine-tuned transformer model on the NER task.

## Conclusion

This notebook successfully demonstrates how to fine-tune a transformer-based model for Named Entity Recognition (NER) using the Hugging Face `transformers` and `datasets` libraries. The fine-tuned model achieves high precision, recall, F1-score, and accuracy, making it suitable for real-world NER tasks.

## Future Work

- **Model Optimization**: Further optimizations, such as hyperparameter tuning, could be performed to improve performance.
- **Extended Dataset**: The model could be trained on a larger and more diverse dataset to improve generalization.
- **Model Deployment**: The fine-tuned model can be deployed to production environments for real-time named entity extraction.

---
