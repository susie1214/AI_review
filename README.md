# 📘 AI 논문 리뷰 포트폴리오

---

✏️ **Word2Vec (2013, Mikolov)**  
📌 핵심 개념  
단어를 벡터 공간에 임베딩하여 의미적 유사성을 수치로 표현  

📌 문제 정의  
One-hot 벡터는 단어 간 의미 관계를 반영하지 못함  

📌 제안 방법  
Skip-gram, CBOW 모델로 주변 단어 맥락 학습  

📌 결과  
유사 단어 벡터 근접  
"king - man + woman = queen" 같은 벡터 연산 가능  

📌 의의  
이후 NLP 임베딩 연구 및 딥러닝 발전의 기초  

---

✏️ **Seq2Seq (2014, Sutskever)**  
📌 핵심 개념  
Encoder-Decoder 구조로 입력 시퀀스를 다른 시퀀스로 변환  

📌 문제 정의  
전통 번역 모델은 긴 문맥과 단어 순서 학습에 한계  

📌 제안 방법  
LSTM Encoder가 입력을 요약 → Decoder가 출력 시퀀스 생성  

📌 결과  
기계 번역 등 자연어 생성 성능 향상  

📌 의의  
Attention/Transformer 탄생의 기반  

---

📖 **Attention Is All You Need (2017, Vaswani)**  
📌 핵심 개념  
Transformer 제안, RNN/CNN을 대체  
Self-Attention으로 문맥 처리  

📌 문제 정의  
RNN: 긴 문맥 처리 한계, 병렬화 어려움  
CNN: 국소적 특징엔 강하나 문맥 전체 파악은 약함  

📌 제안 방법  
Encoder-Decoder 구조 + Multi-head Self Attention  
Positional Encoding으로 순서 정보 보강  

📌 결과  
번역 과제에서 SOTA 달성  
이후 GPT·BERT 등 모든 LLM의 토대가 됨  

📌 의의  
당시 연산 자원 부족 → 대규모 학습 불가  
하지만 현대 LLM 혁신의 뼈대  

---

📖 **BERT (2018, Devlin)**  
📌 핵심 개념  
Bidirectional Encoder Representations로 문맥 이해 강화  

📌 문제 정의  
기존 LM은 좌→우, 우→좌 단방향만 가능  

📌 제안 방법  
Masked LM + Next Sentence Prediction  

📌 결과  
GLUE 등 다수 NLP 벤치마크에서 최고 성능  

📌 의의  
파인튜닝 기반 NLP 혁신 시작  

---

📖 **GPT-3 (2020, Brown)**  
📌 핵심 개념  
초대규모 Transformer LM (175B)  
In-Context Learning 능력  

📌 문제 정의  
소규모 모델은 Zero/Few-shot 학습 한계  

📌 제안 방법  
거대한 LM + Prompt 기반 Few-shot  

📌 결과  
다양한 과제에서 범용적 성능 발휘  

📌 의의  
ChatGPT와 LLM 서비스 시대 개막  

---

📘 **AlexNet (2012, Krizhevsky)**  
📌 핵심 개념  
대규모 CNN으로 이미지 분류 성능 혁신  

📌 문제 정의  
기존 비전 모델은 복잡 이미지 처리에 한계  

📌 제안 방법  
ReLU, Dropout, GPU 병렬 학습  

📌 결과  
ImageNet 대회 압도적 우승  

📌 의의  
딥러닝 컴퓨터비전 붐의 시작  

---

📘 **ResNet (2015, He et al.)**  
📌 핵심 개념  
Residual Connection(잔차 연결) 도입 → 초깊은 네트워크 학습 가능  

📌 문제 정의  
딥러닝 네트워크가 깊어질수록 **기울기 소실/폭발** 문제가 심각해져 성능이 오히려 저하  

📌 제안 방법  
- Residual Block:  
  `F(x) + x` 구조로 입력을 바로 더해줌 (identity shortcut)  
- 네트워크가 깊어져도 **정보/기울기 흐름 보존**  

📌 결과  
- ImageNet 2015 우승 (152층 네트워크)  
- VGG 대비 깊이가 8배, 오차율 대폭 감소  
- Top-5 에러율: 3.57% (당시 SOTA)  

📌 의의  
- 초깊은 CNN 학습의 패러다임을 연 → 이후 모든 비전모델(ResNeXt, DenseNet, EfficientNet 등)의 기본 구조  
- "Residual Learning"은 NLP, 음성, 강화학습 등 전 영역으로 확산  

📌 한계  
- 매우 깊은 네트워크에서 여전히 최적화 난이도 존재  
- 이후 BatchNorm, Bottleneck 구조, Attention 결합으로 보완  

---

📘 **CLIP (2021, Radford)**  
📌 핵심 개념  
텍스트-이미지 대비학습으로 제로샷 분류 가능  

📌 문제 정의  
특정 데이터셋에 한정된 비전 모델 한계  

📌 제안 방법  
대규모 텍스트-이미지 쌍 학습  

📌 결과  
오픈도메인 제로샷 이미지 분류/검색  

📌 의의  
멀티모달 AI의 대표적 전환점  

---

📘 **RAG (2020, Lewis)**  
📌 핵심 개념  
Retrieval-Augmented Generation  

📌 문제 정의  
LLM 환각(hallucination), 최신성 부족  

📌 제안 방법  
외부 검색 결과를 LLM 입력으로 주입  

📌 결과  
QA/지식탐색 성능 향상  

📌 의의  
검색-생성 파이프라인의 표준  

---

📘 **GraphRAG (2024)**  
📌 핵심 개념  
지식 그래프 기반 RAG  

📌 문제 정의  
긴 맥락·대규모 문서에서 단순 검색 한계  

📌 제안 방법  
문서를 그래프로 변환 → 요약·검색 결합  

📌 결과  
복잡 질의 처리 성능 향상  

📌 의의  
최신 RAG 발전 방향  

---

📘 **How transferable are features in deep neural networks? (2014, Yosinski et al.)**  
📌 핵심 개념  
딥러닝의 층별 특징 전이 가능성을 정량화.  
저층 특징은 일반적 → 전이 잘 됨, 중간층은 공적응 붕괴, 상위층은 과제 특이성 강함.  

📌 문제 정의  
- 이미지 분류 네트워크에서 층별 특징이 얼마나 **일반적/특이적**인지?  
- 전이학습 시 어느 층부터 성능 저하가 발생하는지 정량적으로 규명.  

📌 제안 방법  
- 특정 층까지 사전학습 가중치를 복사/동결 → 위층은 랜덤 초기화 후 재학습  
- 자기 전이(BnB) vs. 타 과제 전이(AnB) 비교  
- 미세조정(fine-tuning) 포함한 설정으로 일반성 측정  

📌 결과  
- 하위층(1–2층): 일반적 특징(가보르 필터, 색상 블롭) → 전이성 높음  
- 중간층(3–5층): 공적응 붕괴로 성능 하락 → 미세조정으로 회복 가능  
- 상위층(6–7층): 과제 특이성이 강해 전이성 급감  
- 과제 간 거리(자연물 vs 인공물)가 멀수록 전이 성능 저하가 빠름  
- 그래도 무작위 초기화보다 **사전학습 초기화 + 미세조정**이 항상 우월  

📌 의의  
- 전이학습 실무 가이드라인 제시:  
  - 데이터 적음 → 하위층 고정 + 상위층 학습  
  - 과제 유사 → 상위층 전이도 유효  
  - 과제 상이 → 하위층 위주 전이  
- 사전학습의 **Warm-start 이득**은 거의 모든 층에서 잔존 → 현대 전이학습 기본 이론적 토대  
