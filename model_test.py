import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import json

# 1. 모델 로드
model_path = "./emotion_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. label2id 로드 (id2label 역변환용)
df = pd.read_csv("data/emotion_dataset.csv", sep=",")

texts = df["text"].tolist()

# emotion 컬럼을 label2id로 숫자 변환
with open(f"{model_path}/label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)

df = df[df["emotion"].isin(label2id.keys())].reset_index(drop=True)  # 매핑 안되는 값 제거
texts = df["text"].tolist()
true_labels = df["emotion"].map(label2id).tolist()

# 문자열 라벨이면:
# true_labels = [label2id[l] for l in df["label"].tolist()]

# 4. 배치 추론
preds = []
batch_size = 32

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        preds.extend(batch_preds)

# 5. 정확도 및 리포트
acc = accuracy_score(true_labels, preds)
print(f"정확도: {acc:.4f}")
print(classification_report(true_labels, preds, target_names=[id2label[i] for i in sorted(id2label)]))