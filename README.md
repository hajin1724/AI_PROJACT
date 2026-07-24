# 2026 인공지능 부트캠프 프로젝트
## 주제 : 감정 분류 모델과 LLM이 함께 만드는 텍스트 RPG
### [설명]
플레이어가 한 줄의 텍스트로 행동을 입력하면, 직접 학습시킨 감정 분류 모델이 그 문장에 담긴 감정(기쁨/슬픔/화남/두려움·놀라움)을 판단합니다.
이 분류 결과를 LLM에게 전달하면, LLM은 감정을 게임 속 행동(공격/회피/협력 등)으로 해석하고 그에 맞는 다음 스토리를 실시간으로 생성합니다.
즉, 작은 분류 모델이 "판단"을 맡고 LLM이 "이야기"를 만들어내는 하이브리드 구조로, 플레이어의 선택마다 다른 전개가 이어지는 텍스트 기반 인터랙티브 어드벤처 게임입니다.
Streamlit으로 화면을 구성하고 FastAPI로 분류 모델과 LLM 호출을 처리합니다.

### [사용 기술 & 데이터]
**기술 스택**
- 언어: Python
- 데이터 처리: pandas, numpy
- 자연어 처리: scikit-learn (TfidfVectorizer), konlpy (Okt 형태소 분석기)
- 모델: LogisticRegression (baseline), klue-bert-base 파인튜닝
- 딥러닝 프레임워크: PyTorch, HuggingFace transformers
- 백엔드: FastAPI 
- 프론트엔드: Streamlit 
- LLM 연동: LLM API 
- 개발 환경: VS Code, Python venv, Jupyter Notebook
- 버전 관리: Git, GitHub
- 배포: Docker, Render
- 모델 저장소 관리: Git LFS (420MB 모델 업로드용)

**데이터**
- AI-Hub「감정이 태깅된 자유대화(성인)」— 성인/청소년 라벨링데이터 (텍스트 + 감정 태깅)
- KOTE (Korean Online That-gul Emotions Dataset) — HuggingFace 공개 데이터셋, 44종 세분화 감정 라벨
- 최종 감정 라벨: 기쁨 / 슬픔 / 화남 / 두려움·놀라움 (4종, 언더샘플링으로 클래스 균형 조정)

### [개발 일정]
#### Day 01 (7.20 월) : 데이터 준비 + 시작
- AI-Hub "감정이 태깅된 자유대화(성인)" 데이터셋(성인+청소년) 다운로드 및 승인 신청
- KOTE(Korean Online That-gul Emotions Dataset) 추가 확보 (HuggingFace, 승인 불필요)
- JSON/TSV 전처리 스크립트 작성 → 텍스트+감정라벨 추출
- 두 데이터셋 라벨 통합 (기쁨/없음/놀라움/사랑스러움/화남/두려움/슬픔 7종 → 4종으로 정리: 기쁨/슬픔/화남/두려움·놀라움)
- 클래스 불균형 해소를 위한 언더샘플링 적용
- TF-IDF + 로지스틱회귀 baseline 모델 학습 (정확도 약 40~50%대)
- 형태소 분석기(konlpy Okt) 적용 시도 → 성능 개선 검증 중
- GitHub 저장소 세팅 및 초기 커밋 완료
- 목표 : 데이터 준비 100%, 첫 baseline 모델 학습 성공 ✅ (성능 튜닝은 계속 진행 중)

#### Day 02 (7.21 화) : 결석

#### Day 03 (7.22 수) : API 설계와 스트리밋 구현 및 테스트
- 데이터 병합 파이프라인 버그 수정
- FastAPI ↔ 감정모델 ↔ LLM 통합 (/emotion, /story 엔드포인트)
 --스탯/직업/HP/판정 시스템 설계 및 구현
- Streamlit 프론트엔드 (채팅형 UI, 캐릭터 생성, 사이드바 상태 표시)
- klue-bert-base 파인튜닝 착수 (목표 정확도 70%+)

#### Day 04 (7.23 목) : 모델 마무리 튜닝 및 배포
- klue-bert 모델 재학습 (max_length 64→96, num_train_epochs 3→5, warmup_ratio 추가)
- 정량적 accuracy(46%)는 baseline과 유사한 수준이었으나, 짧고 명확한 감정 표현("슬퍼." 등)에 대한 체감 정확도는 klue-bert가 더 우수함을 확인
- TF-IDF + LogisticRegression
![alt text](image-2.png)
- klue-bert-base (파인튜닝)
![alt text](image-3.png)
- UI/UX 개선
  - 게임오버 시 사망 경위가 표시되지 않던 버그 수정 (session_state에 마지막 스토리 저장 후 게임오버 화면에 출력)
  - 캐릭터 생성 화면에 인트로 스토리 추가
  - 체력 회복 로직 추가 (회복 성공 시 vitality 스탯 기준 최대 체력까지 회복)
  - HP 최대치를 고정값(20)에서 vitality 스탯 연동 방식으로 변경
- 게임 콘텐츠 확장
  - 몬스터 시스템: 10종 몬스터 + 층별 보스(5층 고블린 킹, 10층 리치, 15층 드래곤, 20층 마왕) 추가, LLM이 스토리 맥락에 따라 몬스터 등장 여부 판단
  - 던전 층수 시스템: LLM이 스토리 전개상 층 이동 여부(`floor_change`)를 판단하여 1~20층 진행
  - 직업(20종)·몬스터 이미지 연동 구조 구현
- 배포
  - Dockerfile 작성, FastAPI(내부 전용 127.0.0.1) + Streamlit(외부 노출 0.0.0.0)를 하나의 컨테이너에서 함께 실행하는 구조로 컨테이너화
  - 로컬 Docker 빌드/실행 테스트 완료
  - git-lfs를 이용해 klue-bert 모델(약 420MB) GitHub 저장소에 업로드
  - Render(Web Service, Docker 환경)에 배포 완료
- 목표 : UI 버그 수정, 게임 콘텐츠(몬스터/층수/이미지) 확장, 배포 완료 ✅

#### Day 05 (7.24 금) : 발표 및 마무리
- 발표준비 및 발표
- 보고서 작성 및 최종 결과물 제출

### [배포 링크]
- https://ai-project-co2l.onrender.com
- ⚠️ 무료 인스턴스 특성상 15분 비활성 시 슬립 모드로 전환되며, 재접속 시 최초 로딩에 약 50초 이상 소요될 수 있습니다.

### [주요 기능]
- 텍스트 입력 → 감정 분류(klue-bert) → LLM 기반 행동 판정 및 스토리 생성
- 주사위 판정 시스템 (관련 스탯 + 감정 확신도 기반 성공/실패 결정)
- 직업별 스탯 보너스 및 성공 시 스탯 성장
- 체력(HP) 관리 — 전투 실패 시 감소, 회복 성공 시 최대치까지 회복, 0 이하 시 게임오버
- 던전 층수 진행(1~20층) 및 층별 보스 몬스터 조우
- 직업/몬스터 이미지 표시

### [실행 방법]
**로컬 실행**
```bash
# 터미널 1 - FastAPI 백엔드
uvicorn main:app --reload

# 터미널 2 - Streamlit 프론트엔드
streamlit run app.py
```

**Docker 실행**
```bash
docker build -t dungeon-master .
docker run -p 7860:7860 --env-file .env dungeon-master
```

### [향후 개선 방향]
- 감정 분류 모델의 소수 클래스(두려움/놀라움) recall 개선 (데이터 증강, 추가 하이퍼파라미터 튜닝)
- 몬스터/층 이동 판단 로직의 일관성 검증 및 보강
- 게임 밸런스(스탯-체력 연동 등) QA 플레이테스트 진행
