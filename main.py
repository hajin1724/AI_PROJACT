import json
import os
import random
from pathlib import Path

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from jobs import get_job_bonus

load_dotenv()

# ---------- 설정 ----------
BASE_URL = os.getenv("MLAPI_BASE_URL")
API_KEY = os.getenv("MLAPI_API_KEY")
MODEL_NAME = os.getenv("MLAPI_MODEL", "openai/gpt-5-mini")
EMOTION_THRESHOLD = 0.4
STAT_MAX = 20


if not API_KEY:
    raise RuntimeError("MLAPI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

GAME_WORLD_SETTING = """
이 게임의 배경은 약 20층으로 이루어진 거대한 던전이다.
던전 최하층에는 소유자의 소원을 이루어 준다고 전해지는 전설의 보물이 잠들어 있다.

플레이어는 그 보물을 찾기 위해 던전에 들어온 모험가이며,
현재 위치에서 한 층씩 깊은 곳으로 내려가며 최하층을 목표로 한다.

던전 깊은 곳에는 마왕이 잠들어 있다.
마왕은 최하층의 보물과 관련되어 있으며, 플레이어는 충분한 준비 없이 너무 이른 시점에 마왕을 만나거나 쓰러뜨리면 안 된다.

플레이어는 전사, 기사, 광전사, 검투사, 도적, 궁수, 사냥꾼,
마법사, 소환사, 연금술사, 학자, 점성술사,
수도자, 성기사, 성직자, 탐험가,
음유시인, 상인, 외교관, 무희 중 하나의 직업을 선택해 모험한다.
직업의 특성과 현재 스탯, 행동 판정 결과를 스토리에 자연스럽게 반영한다.
직업과 스탯, 현재 체력, 플레이어의 행동, 행동 판정 성공 여부를 스토리에 자연스럽게 반영한다.

스토리는 탐험, 선택, 전투, NPC와의 만남, 퍼즐, 아이템 획득, 성장 요소를 포함한다.
이전에 밝혀진 사건·장소·인물·아이템·층수와 모순되지 않게 이야기를 이어간다.
아직 도달하지 않은 던전의 정보나 최하층의 결말을 성급하게 공개하지 않는다.
"""



# ---------- 감정 분류기 로드 ----------

# main.py 파일 기준으로 Colab에서 학습한 BERT 모델 폴더 지정
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "emotion_bert_model"

# GPU가 있다면 GPU를 사용하고, 현재처럼 없으면 CPU를 사용한다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 서버 실행 시 BERT 모델과 토크나이저를 한 번만 불러온다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

emotion_model.to(DEVICE)
emotion_model.eval()

# Colab 학습 시 저장한 감정명-클래스 번호 대응표 로드
with open(MODEL_DIR / "label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)

# 예: {"기쁨": 0, "슬픔": 1} -> {0: "기쁨", 1: "슬픔"}
id2label = {
    int(label_id): label
    for label, label_id in label2id.items()
}


def predict_emotion(text: str, threshold: float = EMOTION_THRESHOLD):
    """BERT 모델을 사용해 입력 문장의 감정을 분류한다."""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    # 입력 텐서를 모델이 올라간 장치(CPU 또는 GPU)로 이동
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # 추론 시 gradient 계산 비활성화
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    probabilities = probabilities.cpu().tolist()

    # API 응답용: {"감정명": 확률} 구조 생성
    prob_dict = {
        id2label[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }

    best_id = int(torch.argmax(outputs.logits, dim=1).item())
    max_prob = float(probabilities[best_id])

    # 기존 로직과 동일하게 임계값 미만이면 "없음" 처리
    if max_prob < threshold:
        pred = "없음"
    else:
        pred = id2label[best_id]

    return pred, prob_dict, max_prob



# ---------- 스탯 / 직업 설정 ----------
STAT_NAMES = ["strength", "intelligence", "vitality", "appearance"]


# 텍스트에 이 키워드가 있으면 해당 스탯 체크로 판정 (간단한 규칙 기반 매칭)
STAT_KEYWORDS = {
    "strength": ["힘", "공격", "때리", "베", "밀치", "부수", "들어올리", "싸우", "주먹", "휘두르", "찌르"],
    "intelligence": ["조사", "분석", "추리", "마법", "주문", "알아내", "생각", "계산", "책", "암호", "관찰"],
    "vitality": ["버티", "견디", "달리", "도망", "숨을 참", "체력", "지구력", "회복"],
    "appearance": ["외모", "설득", "협상", "매력", "유혹", "말을 걸", "웃", "미소", "교섭"],
}

# ------------- 밑에 함수로 수정함------------
# def get_relevant_stat(text: str, stats: dict) -> tuple[str, float]:
#     """플레이어 입력 텍스트에서 관련 스탯을 키워드로 추정.
#     매칭되는 키워드가 없으면 4개 스탯 평균으로 종합 판정."""
#     for stat, keywords in STAT_KEYWORDS.items():
#         if any(kw in text for kw in keywords):
#             return stat, float(stats[stat])
#     avg = sum(stats[s] for s in STAT_NAMES) / len(STAT_NAMES)
#     return "종합", avg

def decide_relevant_stat(player_input: str) -> str:
    """
    LLM이 플레이어 행동에 맞는 스탯 하나를 선택한다.
    오류가 나거나 잘못된 값을 반환하면 '종합'을 반환한다.
    """

    system_prompt = (
        "너는 텍스트 RPG의 행동 판정 심판이다. "
        "플레이어 행동을 보고 가장 적절한 스탯 하나만 골라라. "
        "반드시 아래 네 단어 중 하나만 출력해야 한다. "
        "설명, 문장부호, JSON, 마크다운은 절대 출력하지 마라.\n"
        "- strength: 물리 공격, 힘, 무기 사용, 사격, 격투\n"
        "- intelligence: 조사, 추리, 마법, 조준, 제작, 전략, 해독\n"
        "- vitality: 방어, 버티기, 회피, 달리기, 생존, 회복\n"
        "- appearance: 설득, 협상, 연기, 노래, 춤, 유혹, 친화\n"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": player_input},
            ],
            reasoning_effort="minimal",
            verbosity="low",
        )

        chosen_stat = completion.choices[0].message.content.strip().lower()

        valid_stats = {"strength", "intelligence", "vitality", "appearance"}

        if chosen_stat in valid_stats:
            return chosen_stat

    except Exception as e:
        print(f"스탯 판단 LLM 오류: {e}")

    # LLM 오류 또는 형식 불일치 시 기본값
    return "종합"


def roll_dice(confidence: float, stat_value: float, job_bonus: int) -> tuple[int, int, bool]:
    """주사위(1~20)를 굴려서 threshold '이하'가 나오면 성공 (D&D 스킬체크 방식).
    threshold = 관련 스탯/2 + 확신도*10 + 직업 보너스."""
    dice = random.randint(1, 20)
    threshold = round(stat_value / 2 + confidence * 10) + job_bonus
    threshold = max(1, min(19, threshold))
    success = dice <= threshold
    return dice, threshold, success


def generate_story(
    player_input: str,
    emotion: str,
    success: bool,
    context: str,
    relevant_stat: str,
    current_hp: int,
) -> dict:
    system_prompt = (
        "너는 텍스트 기반 RPG 게임의 던전마스터야. 반드시 아래 JSON 형식으로만 응답하고, "
        "그 외의 텍스트(설명, 마크다운 코드블록 등)는 절대 포함하지 마.\n"
        "{\n"
        '  "story": "다음 장면을 2~4문장으로 생생하게 묘사 (판정 실패면 불리하게, 성공이면 유리하게 전개)",\n'
        '  "combat": true 또는 false (이번 상황이 몬스터/적과의 직접적인 전투 상황이면 true, 아니면 false),\n'
        '  "stat_bonus": null 또는 {"stat": "strength|intelligence|vitality|appearance", "amount": 1~2 사이 정수} '
        "(몬스터를 물리치거나 중요한 사건을 성공적으로 해결했을 때만 부여, 그 외엔 null)\n"
        "}"
    )
    user_prompt = (
        f"[게임의 기본 세계관]\n{GAME_WORLD_SETTING.strip()}\n\n"
        f"[현재까지의 스토리 요약]\n"
        f"{context or '게임 시작: 플레이어는 전설의 보물을 찾기 위해 던전 입구에 도착했다. 아직 1층에 진입하지 않았다.'}\n\n"
        f"[플레이어 입력]\n{player_input}\n\n"
        f"[감지된 감정]\n{emotion}\n\n"
        f"[판정에 사용된 스탯]\n{relevant_stat}\n\n"
        f"[행동 판정]\n{'성공' if success else '실패'}\n\n"
        f"[현재 체력]\n{current_hp}\n\n"
        "위 정보와 세계관을 반영해 이전 사건과 모순되지 않도록 다음 장면을 JSON으로 응답해줘."
    )


    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="minimal",
        verbosity="medium",
    )
    raw = completion.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
        story = data.get("story", raw)
        combat = bool(data.get("combat", False))
        stat_bonus = data.get("stat_bonus")
        if stat_bonus and stat_bonus.get("stat") in STAT_NAMES:
            stat_bonus = {
                "stat": stat_bonus["stat"],
                "amount": max(1, min(2, int(stat_bonus.get("amount", 1)))),
            }
        else:
            stat_bonus = None
    except (json.JSONDecodeError, AttributeError, TypeError):
        # LLM이 JSON을 안 지켰을 때를 대비한 안전한 폴백
        story = raw
        combat = False
        stat_bonus = None

    # HP -1은 "전투 상황에서의 판정 실패"일 때만 고정 적용
    hp_change = -1 if (combat and not success) else 0

    return {"story": story, "hp_change": hp_change, "stat_bonus": stat_bonus}


# ---------- FastAPI ----------
app = FastAPI(title="AI 던전마스터 백엔드")


# --- /emotion : 감정 분류만 단독 확인용 ---
class EmotionRequest(BaseModel):
    text: str


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict


@app.post("/emotion", response_model=EmotionResponse)
def classify_emotion(req: EmotionRequest):
    pred, probs, confidence = predict_emotion(req.text)
    return EmotionResponse(emotion=pred, confidence=confidence, probabilities=probs)


# --- /story : 감정분류 -> 스탯 판정 -> LLM 스토리/HP/성장 생성 ---
class StoryRequest(BaseModel):
    player_input: str
    story_context: str = ""
    strength: int = 10
    intelligence: int = 10
    vitality: int = 10
    appearance: int = 10
    current_hp: int = 10       # 현재 체력 (프론트에서 매턴 넘겨받아 갱신 후 되돌려줌)
    job: str = "전사"           # 전사/마법사/수도자/음유시인


class StoryResponse(BaseModel):
    emotion: str
    confidence: float
    relevant_stat: str
    dice_roll: int
    threshold: int
    success: bool
    story: str
    hp_change: int
    current_hp: int
    game_over: bool
    stat_bonus: dict | None
    stats: dict


@app.post("/story", response_model=StoryResponse)
def next_story(req: StoryRequest):
    if not req.player_input.strip():
        raise HTTPException(status_code=400, detail="player_input이 비어있습니다.")

    stats = {
        "strength": req.strength,
        "intelligence": req.intelligence,
        "vitality": req.vitality,
        "appearance": req.appearance,
    }

    emotion, _, confidence = predict_emotion(req.player_input)
    relevant_stat = decide_relevant_stat(req.player_input)

    if relevant_stat == "종합":
        stat_value = sum(stats.values()) / len(stats)
    else:
        stat_value = float(stats[relevant_stat])


    job_bonus = get_job_bonus(req.job, relevant_stat)
    dice, threshold, success = roll_dice(confidence, stat_value, job_bonus)

    result = generate_story(
        req.player_input, emotion, success, req.story_context, relevant_stat, req.current_hp
    )

    # 체력 반영 (0~20 클램프, 0이면 게임오버)
    new_hp = max(0, min(STAT_MAX, req.current_hp + result["hp_change"]))
    game_over = new_hp <= 0

    # 스탯 성장 반영 (최대 20 캡)
    if result["stat_bonus"] and not game_over:
        bonus_stat = result["stat_bonus"]["stat"]
        bonus_amount = result["stat_bonus"]["amount"]
        stats[bonus_stat] = min(STAT_MAX, stats[bonus_stat] + bonus_amount)

    return StoryResponse(
        emotion=emotion,
        confidence=confidence,
        relevant_stat=relevant_stat,
        dice_roll=dice,
        threshold=threshold,
        success=success,
        story=result["story"],
        hp_change=result["hp_change"],
        current_hp=new_hp,
        game_over=game_over,
        stat_bonus=result["stat_bonus"],
        stats=stats,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)