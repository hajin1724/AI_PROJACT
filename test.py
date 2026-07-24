from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "emotion_bert_model"

print("모델 경로:", MODEL_DIR)
print("모델 폴더 존재 여부:", MODEL_DIR.exists())

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

text = "슬퍼."
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

print("logits:", outputs.logits)
print(text)
print("감정별 확률:", probabilities.tolist())
print("예측 클래스 번호:", torch.argmax(probabilities).item())

id2label = {0: "기쁨", 1: "두려움·놀라움", 2: "슬픔", 3: "화남"}
pred_id = torch.argmax(probabilities).item()
print("예측 감정:", id2label[pred_id])