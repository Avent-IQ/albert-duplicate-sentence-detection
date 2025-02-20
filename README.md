# Duplicate Sentence Detection with ALBERT-base-v2

## 📌 Overview

This repository hosts the quantized version of the ALBERT-base-v2 model for Duplicate Sentence Detection. The model is designed to determine whether two sentences convey the same meaning. If they are similar, the model outputs "duplicate" with a confidence score; otherwise, it outputs "not duplicate" with a confidence score. The model has been optimized for efficient deployment while maintaining reasonable accuracy, making it suitable for real-time applications.

## 🏗 Model Details

- **Model Architecture:** ALBERT-base-v2  
- **Task:** Duplicate Sentence Detection  
- **Dataset:** Hugging Face's `quora-question-pairs`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/albert-duplicate-sentence-detection"
model = AlbertForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AlbertTokenizer.from_pretrained(model_name)
```

### Paraphrase Detection Inference

```python
def predict_duplicate(question1, question2, model):
    inputs = tokenizer(question1, question2, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    
    # ✅ Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
        logits = outputs.logits
    
    # ✅ Get prediction
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
 
    # ✅ Output the results
    label_map = {0: "Not Duplicate", 1: "Duplicate"}
    print(f"Q1: {question1}")
    print(f"Q2: {question2}")
    print(f"Prediction: {label_map[prediction]} (Confidence: {probs.max().item():.4f})\n")

# 🔍 Test Example
test_samples = [
    ("How can I learn Python quickly?", "What is the fastest way to learn Python?"),  # Duplicate
    ("What is the capital of India?", "Where is New Delhi located?"),  # Duplicate
    ("How to lose weight fast?", "What is the best programming language to learn?"),  # Not Duplicate
    ("Who is the CEO of Tesla?", "What is the net worth of Elon Musk?"),  # Not Duplicate
    ("What is machine learning?", "How does AI work?"),  # Duplicate
]
for q1, q2 in test_samples:
    predict_duplicate(q1, q2, model)
```

## 📊 Quantized Model Evaluation Results

### 🔥 Evaluation Metrics 🔥

- ✅ **Accuracy:**  0.7215  
- ✅ **Precision:** 0.6497  
- ✅ **Recall:**    0.5440  
- ✅ **F1-score:**  0.5922  

## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle with highly nuanced paraphrases.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different domains and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
