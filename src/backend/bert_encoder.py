from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_bert_embedding(text):
    if not text:
        return [0.0] * 768

    inputs = tokenizer(
        text[:512],
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().tolist()