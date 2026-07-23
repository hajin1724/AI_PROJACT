import random

import requests
import streamlit as st

from jobs import JOB_BONUSES, get_job_image
from monsters import get_monster_image

API_URL = "http://127.0.0.1:8000/story"
JOBS = list(JOB_BONUSES.keys())
STAT_LABELS = {
    "strength": "💪 힘",
    "intelligence": "🧠 지능",
    "vitality": "🏃 체력",
    "appearance": "✨ 외모",
}


st.set_page_config(page_title="AI 던전마스터", page_icon="🎲")
st.title("🎲 AI 던전마스터")

# ---------- 세션 상태 초기화 ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "context" not in st.session_state:
    st.session_state.context = ""

if "stats" not in st.session_state:
    st.session_state.stats = None

if "job" not in st.session_state:
    st.session_state.job = None

if "current_hp" not in st.session_state:
    st.session_state.current_hp = None

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "character_ready" not in st.session_state:
    st.session_state.character_ready = False

if "last_death_story" not in st.session_state:
    st.session_state.last_death_story = None

if "current_floor" not in st.session_state:
    st.session_state.current_floor = 1

def roll_initial_stat():
    """초기 스탯: 3~10 사이 (최대치는 20이지만 시작은 낮게)"""
    return random.randint(3, 10)


def roll_character():
    stats = {
        "strength": roll_initial_stat(),
        "intelligence": roll_initial_stat(),
        "vitality": roll_initial_stat(),
        "appearance": roll_initial_stat(),
    }
    job = random.choice(JOBS)
    return stats, job


def call_story_api(player_input: str, context: str, stats: dict, current_hp: int, job: str, current_floor: int):
    resp = requests.post(
        API_URL,
        json={
            "player_input": player_input,
            "story_context": context,
            "strength": stats["strength"],
            "intelligence": stats["intelligence"],
            "vitality": stats["vitality"],
            "appearance": stats["appearance"],
            "current_hp": current_hp,
            "job": job,
            "current_floor": current_floor,   # 추가
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def reset_game():
    st.session_state.history = []
    st.session_state.context = ""
    st.session_state.stats = None
    st.session_state.job = None
    st.session_state.current_hp = None
    st.session_state.game_over = False
    st.session_state.character_ready = False
    st.session_state.last_death_story = None
    st.session_state.current_floor = 1   # 추가


# ---------- 캐릭터 생성 (게임 시작 전) ----------
if not st.session_state.character_ready:
    st.markdown(
        """
        > 당신은 모험가 입니다.
        > 던전 최하층 20층에 성배가 있다는 소문을 듣고, 성배를 차지하기 위해 던전에 들어왔습니다.
        > 던전에는 수많은 함정과 몬스터가 기다리고 있으며, 던전의 깊은 곳에는 마왕이 잠들어 있습니다.
        > 성배를 차지하고 소원을 이루세요
        """
    )
    st.divider()
    st.subheader("🎲 캐릭터를 생성해주세요")
    st.write("스탯(힘/지능/체력/외모)과 직업을 랜덤으로 정합니다. 직업은 해당 스탯 판정에 보너스를 줍니다.")
    if st.button("🎲 캐릭터 생성하기", type="primary"):
        stats, job = roll_character()
        st.session_state.stats = stats
        st.session_state.job = job
        st.session_state.current_hp = stats["vitality"]  # 시작 체력 = 체력 스탯
        st.session_state.character_ready = True
        st.rerun()
    st.stop()

# ---------- 게임오버 화면 ----------
if st.session_state.game_over:
    st.error("💀 게임 오버")
    if st.session_state.last_death_story:
        st.write(st.session_state.last_death_story)
    if st.button("🔄 새 게임 시작"):
        reset_game()
        st.rerun()
    st.stop()

# ---------- 히스토리 렌더링 ----------
for turn in st.session_state.history:
    if turn["role"] == "player":
        with st.chat_message("user"):
            st.write(turn["text"])
    else:
        with st.chat_message("assistant"):
            st.write(turn["text"])
            if turn.get("monster"):
                st.image(get_monster_image(turn["monster"]), width=200)   # 추가
            badge = "✅ 성공" if turn["success"] else "❌ 실패"
            st.caption(
                f"감정: {turn['emotion']} · 판정 스탯: {turn['relevant_stat']} · "
                f"주사위: {turn['dice']} (목표: {turn['threshold']} 이하) · {badge}"
            )
            if turn.get("hp_change", 0) < 0:
                st.caption(f"💔 체력 {turn['hp_change']}")
            if turn.get("stat_bonus"):
                sb = turn["stat_bonus"]
                st.caption(f"⬆️ {sb['stat']} 스탯 +{sb['amount']}")

# ---------- 입력창 ----------
player_input = st.chat_input("어떻게 할까요?")

if player_input:
    st.session_state.history.append({"role": "player", "text": player_input})
    with st.chat_message("user"):
        st.write(player_input)

    with st.chat_message("assistant"):
        with st.spinner("던전마스터가 생각 중..."):
            try:
                result = call_story_api(
                    player_input,
                    st.session_state.context,
                    st.session_state.stats,
                    st.session_state.current_hp,
                    st.session_state.job,
                    st.session_state.current_floor,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"백엔드 연결 실패: {e}")
                st.stop()

        st.write(result["story"])
        if result.get("monster"):
            st.image(get_monster_image(result["monster"]), width=200)
        badge = "✅ 성공" if result["success"] else "❌ 실패"
        st.caption(
            f"감정: {result['emotion']} · 판정 스탯: {result['relevant_stat']} · "
            f"주사위: {result['dice_roll']} (목표: {result['threshold']} 이하) · {badge}"
        )
        if result["hp_change"] < 0:
            st.caption(f"💔 체력 {result['hp_change']}")
        if result["stat_bonus"]:
            sb = result["stat_bonus"]
            st.caption(f"⬆️ {sb['stat']} 스탯 +{sb['amount']}")

    st.session_state.history.append(
        {
            "role": "dm",
            "text": result["story"],
            "emotion": result["emotion"],
            "relevant_stat": result["relevant_stat"],
            "dice": result["dice_roll"],
            "threshold": result["threshold"],
            "success": result["success"],
            "hp_change": result["hp_change"],
            "stat_bonus": result["stat_bonus"],
            "monster": result["monster"],
        }
    )
    st.session_state.context = result["story"]
    st.session_state.stats = result["stats"]
    st.session_state.current_hp = result["current_hp"]
    st.session_state.current_floor = result["current_floor"] 

    if result["game_over"]:
        st.session_state.game_over = True
        st.session_state.last_death_story = result["story"]
        st.rerun()

# ---------- 사이드바 ----------
with st.sidebar:
    st.header(f"캐릭터 ({st.session_state.job})")
    st.image(get_job_image(st.session_state.job), width=200)
    job_bonuses = JOB_BONUSES[st.session_state.job]
    

    bonus_text = " / ".join(
        f"{STAT_LABELS[stat]} {bonus:+d}"
        for stat, bonus in job_bonuses.items()
    )

    st.caption(f"직업 보너스: {bonus_text}")

    st.metric("❤️ 체력(HP)", f"{st.session_state.current_hp} / {st.session_state.stats['vitality']}")
    st.metric("💪 힘", st.session_state.stats["strength"])
    st.metric("🧠 지능", st.session_state.stats["intelligence"])
    st.metric("🏃 체력(스탯)", st.session_state.stats["vitality"])
    st.metric("✨ 외모", st.session_state.stats["appearance"])
    st.caption("스탯이 높을수록 관련 판정 성공 확률이 올라갑니다. (최대 20)")

    st.divider()
    if st.button("🔄 새 게임 시작"):
        reset_game()
        st.rerun()