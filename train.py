# train.py

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Define label mappings
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Load model
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train and save
trainer.train()
trainer.save_model("./course_feedback_model")
