TEST_IMAGE = "images/default_character.png"

MONSTER_IMAGES = {
    "고블린": "images/goblin.png",
    "오크": "images/ork.png",
    "늑대": TEST_IMAGE,
    "스켈레톤": TEST_IMAGE,
    "슬라임": "images/Slime.png",
    "거미": TEST_IMAGE,
    "좀비": TEST_IMAGE,
    "박쥐괴물": TEST_IMAGE,
    "트롤": TEST_IMAGE,
    "고블린 킹": TEST_IMAGE,
    "리치": TEST_IMAGE,
    "드래곤": TEST_IMAGE,
    "마왕": TEST_IMAGE,
}

DEFAULT_MONSTER_IMAGE = TEST_IMAGE

def get_monster_image(monster):
    return MONSTER_IMAGES.get(monster, DEFAULT_MONSTER_IMAGE)