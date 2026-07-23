# jobs.py

# 직업별 스탯 판정 보정치
# 순서: strength(힘), intelligence(지능), vitality(체력), appearance(외모/매력)
JOB_BONUSES = {
    # 힘 중심 직업
    "전사": {
        "strength": 3,
        "intelligence": -1,
        "vitality": 2,
        "appearance": 0,
    },
    "기사": {
        "strength": 1,
        "intelligence": 1,
        "vitality": 1,
        "appearance": 1,
    },
    "광전사": {
        "strength": 4,
        "intelligence": -2,
        "vitality": 3,
        "appearance": -1,
    },
    "검투사": {
        "strength": 2,
        "intelligence": 1,
        "vitality": 2,
        "appearance": -1,
    },
    "궁수": {
        "strength": 3,
        "intelligence": 0,
        "vitality": 0,
        "appearance": 1,
    },
    "거너": {
        "strength": 2,
        "intelligence": 1,
        "vitality": 0,
        "appearance": 1,
    },
    "무도가": {
        "strength": 2,
        "intelligence": 0,
        "vitality": 2,
        "appearance": 0,
    },
    # 지능 중심 직업
    "마법사": {
        "strength": -2,
        "intelligence": 6,
        "vitality": -2,
        "appearance": 2,
    },
    "도적": {
            "strength": 1,
            "intelligence": 2,
            "vitality": 2,
            "appearance": -1,
        },
    "정령술사": {
        "strength": 0,
        "intelligence": 3,
        "vitality": 0,
        "appearance": 1,
    },
    "연금술사": {
        "strength": 0,
        "intelligence": 3,
        "vitality": 2,
        "appearance": -1,
    },
    "공학자": {
        "strength": 0,
        "intelligence": 4,
        "vitality": 0,
        "appearance": 0,
    },
    "마녀": {
        "strength": -2,
        "intelligence": 4,
        "vitality": -2,
        "appearance": 4,
    },

    # 체력 중심 직업
    "성기사": {
        "strength": 1,
        "intelligence": -1,
        "vitality": 3,
        "appearance": 1,
    },
    "성직자": {
        "strength": 0,
        "intelligence": 1,
        "vitality": 1,
        "appearance": 2,
    },
    "악마사냥꾼": {
        "strength": 2,
        "intelligence": 1,
        "vitality": 3,
        "appearance": -2,
    },

    # 매력 중심 직업
    "음유시인": {
        "strength": 0,
        "intelligence": 0,
        "vitality": 0,
        "appearance": 4,
    },
    "상인": {
        "strength": -1,
        "intelligence": 2,
        "vitality": 1,
        "appearance": 1,
    },
    "무희": {
        "strength": 1,
        "intelligence": -1,
        "vitality": 2,
        "appearance": 2,
    },
    # 특수 직업
    "용사": {
            "strength": 3,
            "intelligence": 3,
            "vitality": 3,
            "appearance": 3,
        },
}


def get_job_bonus(job: str, relevant_stat: str) -> int:
    """직업과 현재 판정 스탯에 맞는 보정치를 반환한다."""

    if relevant_stat == "종합":
        return 0

    normalized_job = job.strip()
    return JOB_BONUSES.get(normalized_job, {}).get(relevant_stat, 0)

JOB_IMAGES = {
    #"전사": "images/warrior.png",
    #"마법사": "images/mage.png",
    # 그린 것부터 하나씩 채우기
}

DEFAULT_IMAGE = "images/default_character.png"

def get_job_image(job):
    return JOB_IMAGES.get(job, DEFAULT_IMAGE)