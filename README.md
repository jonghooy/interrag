<<<<<<< HEAD
# FAQ 검색 및 문의 요약 시스템

LangChain과 Streamlit을 사용하여 FAQ 검색 및 문의 요약을 자동화하는 시스템입니다.

## 저장소 정보

- GitHub: [https://github.com/jonghooy/interrag](https://github.com/jonghooy/interrag)

## 주요 기능

- FAQ 데이터베이스에서 유사한 질문 검색
- 검색 결과를 기반으로 한 자동 답변 생성
- 검색 결과 및 문의 내용 저장
- 한국 시간 기준 타임스탬프 자동 기록

## 시스템 요구사항

- Python 3.8 이상
- OpenAI API 키

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/jonghooy/interrag.git
cd interrag
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
- `.env` 파일 생성:
```bash
touch .env
```
- `.env` 파일에 OpenAI API 키 설정:
```
OPENAI_API_KEY=your_api_key_here
```

## 데이터 준비

1. FAQ 데이터 생성:
```bash
python create_faq.py
```
- 이 명령어는 `data/faq.xlsx` 파일을 생성합니다
- 기본 샘플 FAQ 데이터가 포함됩니다

## 실행 방법

1. Streamlit 앱 실행:
```bash
streamlit run app.py
```

2. 웹 브라우저 접속:
- 자동으로 브라우저가 열립니다
- 또는 수동으로 `http://localhost:8501` 접속

## 사용 방법

1. 웹 인터페이스에서 문의 내용 입력
2. "검색" 버튼 클릭
3. 검색 결과 확인
4. 결과는 자동으로 `data/results.xlsx`에 저장됨

## 데이터 구조

- `data/faq.xlsx`: FAQ 데이터베이스
  - question: 질문
  - answer: 답변

- `data/results.xlsx`: 검색 결과 저장
  - query: 사용자 문의
  - response: 시스템 응답
  - timestamp: 한국 시간 기준 기록 시간

## 문제 해결

1. 패키지 설치 오류:
```bash
pip install -r requirements.txt --upgrade
```

2. API 키 관련 오류:
- `.env` 파일의 API 키 확인
- OpenAI API 키가 유효한지 확인

3. FAQ 데이터 오류:
```bash
python create_faq.py
```

## 주의사항

- OpenAI API 키는 보안을 위해 `.env` 파일에만 저장
- FAQ 데이터는 Excel 파일 형식으로 저장
- 검색 결과는 자동으로 `results.xlsx`에 누적 저장

## 라이선스

MIT License

## 문의

- GitHub Issues: [https://github.com/jonghooy/interrag/issues](https://github.com/jonghooy/interrag/issues) 
=======
# interrag
>>>>>>> 9a406b3bd08595b33327b102bb03d71c496c8f71
