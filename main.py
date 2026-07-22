import json
import os
import pickle
import random
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from konlpy.tag import Okt
import uvicorn

load_dotenv()

# ---------- 설정 ----------
BASE_URL = os.getenv("MLAPI_BASE_URL")
API_KEY = os.getenv("MLAPI_API_KEY")
MODEL_NAME = os.getenv("MLAPI_MODEL", "openai/gpt-5-mini")
EMOTION_THRESHOLD = 0.35
STAT_MAX = 20
JOB_BONUS_AMOUNT = 2

if not API_KEY:
    raise RuntimeError("MLAPI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---------- 감정 분류기 로드 ----------
okt = Okt()


def tokenize(text: str):
    return okt.morphs(text)


# vectorizer.pkl은 노트북(__main__)에서 학습되어 tokenizer가 __main__.tokenize를
# 참조하도록 저장되어 있음. uvicorn으로 실행하면 이 프로세스의 __main__이
# main.py 자신이 아니게 되므로, pickle.load 전에 __main__에 tokenize를 직접 등록해줘야 함.
sys.modules["__main__"].tokenize = tokenize

with open("emotion_model.pkl", "rb") as f:
    emotion_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict_emotion(text: str, threshold: float = EMOTION_THRESHOLD):
    vec = vectorizer.transform([text])
    proba = emotion_model.predict_proba(vec)[0]
    prob_dict = {label: float(p) for label, p in zip(emotion_model.classes_, proba)}

    max_prob = max(proba)
    if max_prob < threshold:
        pred = "없음"
    else:
        pred = emotion_model.classes_[proba.argmax()]

    return pred, prob_dict, float(max_prob)


# ---------- 스탯 / 직업 설정 ----------
STAT_NAMES = ["strength", "intelligence", "vitality", "appearance"]

# 직업 -> 어떤 스탯 체크에 보너스를 주는지
JOB_TO_STAT = {
    "전사": "strength",
    "마법사": "intelligence",
    "수도자": "vitality",
    "음유시인": "appearance",
}

# 텍스트에 이 키워드가 있으면 해당 스탯 체크로 판정 (간단한 규칙 기반 매칭)
STAT_KEYWORDS = {
    "strength": ["힘", "공격", "때리", "베", "밀치", "부수", "들어올리", "싸우", "주먹", "휘두르", "찌르"],
    "intelligence": ["조사", "분석", "추리", "마법", "주문", "알아내", "생각", "계산", "책", "암호", "관찰"],
    "vitality": ["버티", "견디", "달리", "도망", "숨을 참", "체력", "지구력", "회복"],
    "appearance": ["외모", "설득", "협상", "매력", "유혹", "말을 걸", "웃", "미소", "교섭"],
}


def get_relevant_stat(text: str, stats: dict) -> tuple[str, float]:
    """플레이어 입력 텍스트에서 관련 스탯을 키워드로 추정.
    매칭되는 키워드가 없으면 4개 스탯 평균으로 종합 판정."""
    for stat, keywords in STAT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return stat, float(stats[stat])
    avg = sum(stats[s] for s in STAT_NAMES) / len(STAT_NAMES)
    return "종합", avg


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
        '  "hp_change": 정수 (몬스터/적과의 전투에서 판정 실패로 피해를 입었을 때만 -1~-5 사이 음수, 그 외에는 0),\n'
        '  "stat_bonus": null 또는 {"stat": "strength|intelligence|vitality|appearance", "amount": 1~2 사이 정수} '
        "(몬스터를 물리치거나 중요한 사건을 성공적으로 해결했을 때만 부여, 그 외엔 null)\n"
        "}"
    )
    user_prompt = (
        f"[이전 상황]\n{context or '(게임 시작)'}\n\n"
        f"[플레이어 입력]\n{player_input}\n\n"
        f"[감지된 감정]\n{emotion}\n\n"
        f"[판정에 사용된 스탯]\n{relevant_stat}\n\n"
        f"[행동 판정]\n{'성공' if success else '실패'}\n\n"
        f"[현재 체력]\n{current_hp}\n\n"
        "위 정보를 반영해서 다음 장면을 JSON으로 응답해줘."
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
        hp_change = int(data.get("hp_change") or 0)
        hp_change = max(-5, min(0, hp_change))  # 전투 피해만 반영, 힐은 없음
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
        hp_change = 0
        stat_bonus = None

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
    relevant_stat, stat_value = get_relevant_stat(req.player_input, stats)

    job_bonus = JOB_BONUS_AMOUNT if JOB_TO_STAT.get(req.job) == relevant_stat else 0
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