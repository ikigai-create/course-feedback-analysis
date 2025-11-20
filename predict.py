# predict.py

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./course_feedback_model")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Sample feedback
sample_texts = [
    "The instructor was very clear and helpful.",
    "Assignments were confusing and stressful."
]

# Tokenize and move to device
encodings = tokenizer(sample_texts, truncation=True, padding=True, return_tensors="pt")
encodings = {k: v.to(model.device) for k, v in encodings.items()}

# Predict
outputs = model(**encodings)
preds = torch.argmax(outputs.logits, dim=1)

# Map predictions to labels
predicted_sentiments = [model.config.id2label[p.item()] for p in preds]

# Print results
for text, sentiment in zip(sample_texts, predicted_sentiments):
    print(f"Feedback: {text}\nPrediction: {sentiment}\n")
