# app.py (최종 버전 - 좌우 분할 및 FAQ 처리)

import streamlit as st
import os
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime
import tiktoken
import pandas as pd
import json # LLM 응답 파싱용 (필요시 대비)
import re # load_template_dataframe 내에서 사용
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytz
import sys

# 현재 디렉토리를 Python 경로에 명시적으로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 환경 변수 로드
load_dotenv()

# 유틸리티 함수 임포트 (utils.py 파일이 필요합니다)
try:
    from utils import (
        load_embedding_model, load_faiss_index_and_ids,
        load_template_dataframe, load_faq_dataframe, # 함수 분리됨
        call_openai_api, retrieve_candidates,
        load_prompt, save_prompt, parse_llm_recommendation_text,
        debug_print_to_file # 디버그 파일 저장 함수 추가
    )
except ImportError:
    st.error("오류: 'utils.py' 파일을 찾을 수 없습니다. app.py와 같은 디렉토리에 있는지 확인하세요.")
    st.stop()

# --- 페이지 설정 (다른 st 명령어보다 먼저 호출) ---
st.set_page_config(
    page_title="인터파크 템플릿/FAQ 추천봇",
    layout="wide", # 넓게 설정
    initial_sidebar_state="expanded"
)

# debug 디렉토리 생성 로직
debug_dir = 'debug'
if not os.path.exists(debug_dir):
    try:
        os.makedirs(debug_dir)
        print(f"디버그 디렉토리 생성됨: {debug_dir}")
    except OSError as e:
        print(f"디버그 디렉토리 생성 실패: {debug_dir}, 오류: {e}")
        st.warning(f"디버그 디렉토리({debug_dir}) 생성에 실패했습니다. 권한을 확인하세요.")

def debug_print_to_file(title, content, filename=None):
    """디버그 내용을 파일에 기록합니다."""
    if filename is None:
        filename = f"debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    filepath = os.path.join(debug_dir, filename)
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*30}\n{title}\n{'='*30}\n")
        f.write(str(content))

def debug_template_data():
    """문제 파악을 위해 템플릿 데이터를 JSON 파일로 저장"""
    problem_ids = [580, 637]
    debug_data = {}
    
    for pid in problem_ids:
        rows = agent_template_df[agent_template_df['ENTITY_ID'] == pid]
        if not rows.empty:
            row = rows.iloc[0]
            content = row.get('CONTENT', '내용 없음')
            cleaned_content = row.get('CLEANED_CONTENT', '정제된 내용 없음')
            
            debug_data[f"ID_{pid}"] = {
                "entity_id": pid,
                "template_name": row.get('TEMPLATE_NAME', '이름 없음'),
                "content_length": len(str(content)),
                "cleaned_content_length": len(str(cleaned_content)),
                "content": str(content),
                "cleaned_content": str(cleaned_content)
            }
    
    # JSON 저장
    debug_file = os.path.join(debug_dir, "debug_template_data.json")
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    
    return f"템플릿 데이터 디버깅 파일이 생성되었습니다: {debug_file}"

# --- 로깅 설정 ---
log_dir = 'log'
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError as e:
        logging.error(f"로그 디렉토리 생성 실패: {log_dir}, 오류: {e}", exc_info=True)
        st.error(f"로그 디렉토리({log_dir}) 생성에 실패했습니다. 권한을 확인하세요.")
        st.stop()

log_filename = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = logging.getLogger('StreamlitAppLogger')
logger.setLevel(logging.DEBUG) # 디버깅 위해 DEBUG 레벨 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 핸들러 중복 추가 방지 로직 개선
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    try:
        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setLevel(logging.DEBUG) # 파일 핸들러도 DEBUG 레벨 설정
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        print(f"로그 파일 생성됨: {log_filename}") # 콘솔에도 간단히 알림
    except Exception as e:
        logging.error(f"파일 핸들러 설정 오류: {e}", exc_info=True)
        st.error(f"로그 파일 핸들러 설정 중 오류 발생: {e}")

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # 콘솔은 INFO 레벨 유지
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # logger.info("콘솔 핸들러 추가 완료.") # 로거 사용 전 print 사용 권장

print("로거 설정 완료.") # 로거 사용 전 print
logger.info("="*20 + " Streamlit 앱 시작 / 재실행 " + "="*20)
# --- 로깅 설정 끝 ---

# --- 기본 설정값 ---
TEMPLATE_FILEPATH = 'structured_templates_250412_1.txt' # 실제 사용하는 파일명으로 변경
TEMPLATE_IDS_FILENAME = 'template_ids.npy'
TEMPLATE_INDEX_FILENAME = 'faiss_index.index'
FAQ_TEMPLATE_FILEPATH = 'User_FAQ_templates.txt'
FAQ_IDS_FILENAME = 'faq_ids.npy'
FAQ_INDEX_FILENAME = 'faq_faiss_index.index'
EMBEDDING_MODEL_NAME = 'jhgan/ko-sbert-sts'
PROMPT_DIR = 'prompts'
QUERY_PROMPT_FILE = os.path.join(PROMPT_DIR, 'query_enhancement_prompt.txt')
RECOMMEND_PROMPT_FILE = os.path.join(PROMPT_DIR, 'final_recommendation_prompt.txt')
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8192 # LLM 답변 최대 길이
CONTEXT_LENGTHS = {"gpt-4o": 128000, "gpt-4-turbo": 128000, "gpt-3.5-turbo": 16385}
DEFAULT_NUM_CANDIDATES = 20 # 기본 검색 후보 수
DEFAULT_NUM_RECOMMENDATIONS = 3 # 기본 최종 추천 수

# API 키 로드
load_dotenv(); openai_api_key = os.getenv(OPENAI_API_KEY_ENV_VAR);
if not openai_api_key: st.error("OpenAI API 키를 .env 파일에 설정해주세요."); logger.error("OpenAI API 키 환경 변수 로드 실패."); st.stop()
else: openai.api_key = openai_api_key; logger.info("OpenAI API 키 로드 완료.")

# --- 리소스 로드 (캐싱) ---
@st.cache_resource
def load_all_resources():
    logger.info("모든 리소스 로드 시도...")
    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    agent_index, agent_ids = load_faiss_index_and_ids(TEMPLATE_INDEX_FILENAME, TEMPLATE_IDS_FILENAME)
    agent_df = load_template_dataframe(TEMPLATE_FILEPATH) # 상담사 템플릿 로드
    faq_index, faq_ids = load_faiss_index_and_ids(FAQ_INDEX_FILENAME, FAQ_IDS_FILENAME)
    faq_df = load_faq_dataframe(FAQ_TEMPLATE_FILEPATH) # FAQ 전용 로드 함수 사용
    encoding = None
    try: encoding = tiktoken.get_encoding("cl100k_base"); logger.info("Tiktoken 인코더 로드 완료.")
    except Exception as e: logger.error(f"Tiktoken 인코더 로드 실패: {e}")
    return model, agent_index, agent_ids, agent_df, faq_index, faq_ids, faq_df, encoding

embedding_model, \
agent_faiss_index, agent_template_ids, agent_template_df, \
faq_faiss_index, faq_template_ids, faq_template_df, \
tiktoken_encoding = load_all_resources()

# 필수 리소스 로드 확인
required_resources = [embedding_model, agent_faiss_index, agent_template_ids, agent_template_df,
                      faq_faiss_index, faq_template_ids, faq_template_df]
if not all(res is not None for res in required_resources):
    st.error("필수 리소스 로드에 실패했습니다. 로그를 확인하거나 prepare_data*.py 스크립트를 실행하세요.")
    resource_names = ["Embedding Model", "Agent Index", "Agent IDs", "Agent DF", "FAQ Index", "FAQ IDs", "FAQ DF"]
    for name, res in zip(resource_names, required_resources):
        if res is None: logger.error(f"앱 중지: {name} 로드 실패")
    st.stop()
else: logger.info("모든 필수 리소스 로드 확인 완료.")

# --- Streamlit 세션 상태 초기화 ---
default_session_state = {
    "history": [], "user_query": "", "enhanced_query": "",
    "candidates_agent": [], "candidates_faq": [],
    "recommendations_agent": [], "recommendations_faq": [],
    "processing_done": False,
    "selected_candidate_agent_key": None, "selected_candidate_faq_key": None,
    "last_processed_query": ""
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- 사이드바 설정 ---
st.sidebar.title("⚙️ 설정")
llm_model_options = list(CONTEXT_LENGTHS.keys())
selected_llm_model = st.sidebar.selectbox("LLM 모델", llm_model_options, index=llm_model_options.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in llm_model_options else 0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens (답변)", 50, 16384, DEFAULT_MAX_TOKENS, 10) # 사용자 수정값 반영
context_length = CONTEXT_LENGTHS.get(selected_llm_model, 0)
st.sidebar.caption(f"선택된 모델 컨텍스트 길이: {context_length} 토큰")
num_candidates = st.sidebar.number_input("검색할 후보 수 (공통)", 3, 20, DEFAULT_NUM_CANDIDATES, 1, key="num_cand")
num_recommendations = st.sidebar.number_input("최종 추천 수 (공통)", 1, 5, DEFAULT_NUM_RECOMMENDATIONS, 1, key="num_rec")
# 프롬프트 관리
query_prompt_content = load_prompt(QUERY_PROMPT_FILE) or "# 기본 쿼리 향상 프롬프트..."
edited_query_prompt = st.sidebar.text_area("쿼리 향상 프롬프트", value=query_prompt_content, height=150, key="query_prompt_editor")
if st.sidebar.button("쿼리 프롬프트 저장"):
    if save_prompt(QUERY_PROMPT_FILE, edited_query_prompt): st.sidebar.success("저장 완료")
    else: st.sidebar.error("저장 실패")
recommend_prompt_content = load_prompt(RECOMMEND_PROMPT_FILE) or "# 기본 최종 추천 프롬프트..."
edited_recommend_prompt = st.sidebar.text_area("최종 추천 프롬프트", value=recommend_prompt_content, height=300, key="rec_prompt_editor")
if st.sidebar.button("추천 프롬프트 저장"):
    if save_prompt(RECOMMEND_PROMPT_FILE, edited_recommend_prompt): st.sidebar.success("저장 완료")
    else: st.sidebar.error("저장 실패")
# 초기화 버튼
if st.sidebar.button("대화 내용 및 결과 초기화"):
    logger.info("대화 내용 및 결과 초기화 요청.")
    for key in default_session_state: st.session_state[key] = default_session_state[key]
    st.rerun()

# 사이드바에 디버깅 섹션 추가
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 디버깅 도구")
debug_id = st.sidebar.number_input("템플릿 ID로 직접 검색", min_value=1, value=580) # max_value 제거 또는 적절히 설정

if st.sidebar.button("상담사 템플릿 검색"): # 버튼 이름 명확화
    # --- !!! 변수 이름 수정: agent_template_df -> agent_template_df !!! ---
    rows = agent_template_df[agent_template_df['ENTITY_ID'] == debug_id]
    if not rows.empty:
        debug_data = rows.iloc[0].to_dict()
        st.sidebar.success(f"ID {debug_id} 찾음!")
        # 표시할 정보 선택 (필요시 조정)
        st.sidebar.json({
            "ENTITY_ID": debug_data.get('ENTITY_ID'),
            "TEMPLATE_NAME": debug_data.get('TEMPLATE_NAME'), # agent_template_df에는 TEMPLATE_NAME 있음
            "CONTENT_Length": len(str(debug_data.get('CONTENT', ''))),
            "CLEANED_CONTENT_Length": len(str(debug_data.get('CLEANED_CONTENT', '')))
        })

        # 내용 다운로드 (원본 CONTENT 기준)
        content = debug_data.get('CONTENT', '')
        # --- !!! 변수 이름 수정 !!! ---
        debug_content_filename = f"debug_agent_template_{debug_id}_content.txt"
        try:
            with open(debug_content_filename, "w", encoding="utf-8") as f:
                f.write(content)
            with open(debug_content_filename, "rb") as f:
                st.sidebar.download_button(
                    label=f"ID {debug_id} 내용 다운로드",
                    data=f,
                    file_name=debug_content_filename,
                    mime="text/plain"
                )
            # 임시 파일 삭제 (선택 사항)
            # if os.path.exists(debug_content_filename): os.remove(debug_content_filename)
        except Exception as e:
            st.sidebar.error(f"파일 생성/다운로드 오류: {e}")

    else:
        st.sidebar.error(f"상담사 템플릿에서 ID {debug_id}를 찾을 수 없습니다.")


# FAQ 데이터 디버깅 버튼 추가 (선택 사항)
debug_faq_id = st.sidebar.number_input("FAQ ID로 직접 검색", min_value=1, value=1117)
if st.sidebar.button("FAQ 검색"):
    # --- !!! faq_template_df 사용 !!! ---
    rows_faq = faq_template_df[faq_template_df['ENTITY_ID'] == debug_faq_id]
    if not rows_faq.empty:
        debug_data_faq = rows_faq.iloc[0].to_dict()
        st.sidebar.success(f"FAQ ID {debug_faq_id} 찾음!")
        # FAQ 컬럼명 사용 (TITLE, DESCRIPTION 등)
        st.sidebar.json({
            "ENTITY_ID": debug_data_faq.get('ENTITY_ID'),
            "TITLE": debug_data_faq.get('TITLE'), # FAQ는 TITLE
            "DESCRIPTION_Length": len(str(debug_data_faq.get('DESCRIPTION', ''))),
            "CLEANED_DESCRIPTION_Length": len(str(debug_data_faq.get('CLEANED_DESCRIPTION', '')))
        })
        # 내용 다운로드 (원본 DESCRIPTION 기준)
        content_faq = debug_data_faq.get('DESCRIPTION', '')
        debug_faq_content_filename = f"debug_faq_{debug_faq_id}_content.txt"
        try:
            with open(debug_faq_content_filename, "w", encoding="utf-8") as f:
                f.write(content_faq)
            with open(debug_faq_content_filename, "rb") as f:
                st.sidebar.download_button(
                    label=f"FAQ ID {debug_faq_id} 내용 다운로드",
                    data=f,
                    file_name=debug_faq_content_filename,
                    mime="text/plain"
                )
            # if os.path.exists(debug_faq_content_filename): os.remove(debug_faq_content_filename)
        except Exception as e:
            st.sidebar.error(f"FAQ 파일 생성/다운로드 오류: {e}")
    else:
        st.sidebar.error(f"FAQ에서 ID {debug_faq_id}를 찾을 수 없습니다.")

# 전체 템플릿 데이터 디버깅 버튼
if st.sidebar.button("전체 템플릿 데이터 디버깅"):
    debug_result = debug_template_data()
    st.sidebar.success(debug_result)
    
    # 디버그 파일 다운로드 링크 제공
    debug_file = os.path.join(debug_dir, "debug_template_data.json")
    with open(debug_file, "rb") as f:
        st.sidebar.download_button(
            label="디버그 데이터 다운로드",
            data=f,
            file_name="debug_template_data.json",
            mime="application/json"
        )

# LLM 프롬프트/응답 다운로드 버튼
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 LLM 통신 디버깅")

debug_prompt_file = os.path.join(debug_dir, "debug_prompt.txt")
if os.path.exists(debug_prompt_file):
    with open(debug_prompt_file, "rb") as f:
        st.sidebar.download_button(
            label="최신 프롬프트 다운로드",
            data=f,
            file_name="debug_prompt.txt",
            mime="text/plain"
        )

debug_response_file = os.path.join(debug_dir, "debug_response.txt")
if os.path.exists(debug_response_file):
    with open(debug_response_file, "rb") as f:
        st.sidebar.download_button(
            label="최신 응답 다운로드",
            data=f,
            file_name="debug_response.txt",
            mime="text/plain"
        )

# 원본 파일 확인 버튼
if st.sidebar.button("원본 템플릿 파일 검증"):
    if os.path.exists(TEMPLATE_FILEPATH):
        with open(TEMPLATE_FILEPATH, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        # 원본 파일 내용 저장
        with open("original_template_file.txt", "w", encoding="utf-8") as f:
            f.write(file_content)
        
        # 다운로드 링크 제공
        with open("original_template_file.txt", "rb") as f:
            st.sidebar.download_button(
                label="원본 템플릿 파일 다운로드",
                data=f,
                file_name="original_template_file.txt",
                mime="text/plain"
            )
        
        st.sidebar.success("원본 템플릿 파일 검증 완료")
    else:
        st.sidebar.error(f"템플릿 파일을 찾을 수 없음: {TEMPLATE_FILEPATH}")

# --- 메인 화면 UI ---
st.title("💬 인터파크 템플릿/FAQ 추천봇 (RAG Demo)")
st.caption(f"Embedding: {EMBEDDING_MODEL_NAME} | Index: Faiss | LLM: {selected_llm_model}")

# --- 메인 로직 ---
user_query_input = st.chat_input("고객 문의 내용을 입력하세요:")

if user_query_input and user_query_input != st.session_state.get("last_processed_query", ""):
    logger.info(f"새 사용자 쿼리 수신: \"{user_query_input}\"")
    st.session_state.user_query = user_query_input
    st.session_state.enhanced_query = ""
    st.session_state.candidates_agent = []
    st.session_state.candidates_faq = []
    st.session_state.recommendations_agent = []
    st.session_state.recommendations_faq = []
    st.session_state.processing_done = False
    st.session_state.selected_candidate_agent_key = None
    st.session_state.selected_candidate_faq_key = None
    st.session_state.last_processed_query = user_query_input

    # 1. 쿼리 향상
    with st.spinner("LLM으로 검색 쿼리 최적화 중..."):
        current_query_prompt = edited_query_prompt.format(user_query=st.session_state.user_query)
        enhanced_query = call_openai_api(current_query_prompt, selected_llm_model, temperature, max_tokens)
        if enhanced_query: st.session_state.enhanced_query = enhanced_query; logger.info(f"향상된 쿼리 생성: \"{enhanced_query}\"")
        else: logger.error("쿼리 향상 실패."); st.error("쿼리 향상에 실패했습니다."); st.session_state.processing_done = True

    # 2. 병렬 검색
    if st.session_state.enhanced_query and not st.session_state.processing_done:
        with st.spinner(f"상담사 템플릿 & FAQ 검색 중... (상위 {num_candidates}개)"):
            # 2-A: 상담사 템플릿 검색
            candidates_agent = retrieve_candidates(st.session_state.enhanced_query, embedding_model, agent_faiss_index, agent_template_ids, agent_template_df, k=num_candidates)
            st.session_state.candidates_agent = candidates_agent
            if candidates_agent:
                # retrieve_candidates 반환값의 'TEMPLATE_NAME' 사용
                first_key = f"1. {candidates_agent[0].get('TEMPLATE_NAME', '이름 없음')} (ID: {candidates_agent[0]['entity_id']}, Score: {candidates_agent[0]['score']:.4f})"
                st.session_state.selected_candidate_agent_key = first_key
            else: st.session_state.selected_candidate_agent_key = None; logger.warning("상담사 템플릿 검색 결과 없음.")

            # 2-B: FAQ 검색
            candidates_faq = retrieve_candidates(st.session_state.enhanced_query, embedding_model, faq_faiss_index, faq_template_ids, faq_template_df, k=num_candidates)
            st.session_state.candidates_faq = candidates_faq
            if candidates_faq:
                 # retrieve_candidates 반환값의 'TITLE' 사용 (utils.py에서 해당 키로 반환한다고 가정)
                 first_key = f"1. {candidates_faq[0].get('TITLE', '제목 없음')} (ID: {candidates_faq[0]['entity_id']}, Score: {candidates_faq[0]['score']:.4f})"
                 st.session_state.selected_candidate_faq_key = first_key
            else: st.session_state.selected_candidate_faq_key = None; logger.warning("FAQ 검색 결과 없음.")

    # 3. 병렬 최종 추천
    if not st.session_state.processing_done:
        # 3-A: 상담사 템플릿 추천
        if st.session_state.candidates_agent:
            with st.spinner(f"LLM에게 상담사 템플릿 추천 요청 중..."):
                candidate_list_str = ""
                for i, cand in enumerate(st.session_state.candidates_agent):
                    # retrieve_candidates 반환값의 'CONTENT' 사용
                    content_preview = cand.get('CONTENT', '')[:500] + "..." if len(cand.get('CONTENT', '')) > 500 else cand.get('CONTENT', '')
                    candidate_list_str += f"\n--- 템플릿 {i+1} ---\nENTITY_ID: {cand['entity_id']}\nTEMPLATE_NAME: {cand.get('TEMPLATE_NAME', '이름 없음')}\nCONTENT:\n{content_preview}\n"
                current_recommend_prompt = edited_recommend_prompt.format( user_query=st.session_state.user_query, enhanced_query=st.session_state.enhanced_query, candidate_count=len(st.session_state.candidates_agent), candidate_list_str=candidate_list_str, num_recommendations=num_recommendations )
                # 디버깅: 최종 프롬프트 파일 저장
                debug_print_to_file("Agent Recommend Prompt", current_recommend_prompt, "debug_agent_prompt.txt")
                # 토큰 계산
                prompt_tokens = 0
                if tiktoken_encoding:
                    try: prompt_tokens = len(tiktoken_encoding.encode(current_recommend_prompt)); logger.info(f"상담사 추천 프롬프트 토큰 수: {prompt_tokens}")
                    except Exception as enc_e: logger.error(f"Tiktoken 인코딩 오류: {enc_e}")
                # LLM 호출
                llm_response_text = call_openai_api(current_recommend_prompt, selected_llm_model, temperature, max_tokens)
                if llm_response_text:
                    recommendations = parse_llm_recommendation_text(llm_response_text)
                    st.session_state.recommendations_agent = recommendations if isinstance(recommendations, list) else []
                    if not st.session_state.recommendations_agent: logger.error("상담사 템플릿 LLM 응답 파싱 실패."); st.error("상담사 템플릿 LLM 응답 파싱 실패.")
                else: st.session_state.recommendations_agent = []; logger.error("상담사 템플릿 LLM 결과 수신 실패."); st.error("상담사 템플릿 LLM 결과 수신 실패.")
        else: st.session_state.recommendations_agent = []

        # 3-B: FAQ 추천
        if st.session_state.candidates_faq:
            with st.spinner(f"LLM에게 FAQ 추천 요청 중..."):
                candidate_list_str = ""
                for i, cand in enumerate(st.session_state.candidates_faq):
                    # retrieve_candidates 반환값의 'TITLE', 'DESCRIPTION' 사용
                    faq_title = cand.get('TITLE', '제목 없음')
                    faq_content = cand.get('DESCRIPTION', '') # FAQ는 DESCRIPTION이 원본 내용
                    content_preview = faq_content[:500] + "..." if len(faq_content) > 500 else faq_content
                    candidate_list_str += f"\n--- FAQ {i+1} ---\nENTITY_ID: {cand['entity_id']}\nTITLE: {faq_title}\nDESCRIPTION:\n{content_preview}\n"
                current_recommend_prompt = edited_recommend_prompt.format( user_query=st.session_state.user_query, enhanced_query=st.session_state.enhanced_query, candidate_count=len(st.session_state.candidates_faq), candidate_list_str=candidate_list_str, num_recommendations=num_recommendations )
                # 디버깅: 최종 프롬프트 파일 저장
                debug_print_to_file("FAQ Recommend Prompt", current_recommend_prompt, "debug_faq_prompt.txt")
                # 토큰 계산
                prompt_tokens = 0
                if tiktoken_encoding:
                    try: prompt_tokens = len(tiktoken_encoding.encode(current_recommend_prompt)); logger.info(f"FAQ 추천 프롬프트 토큰 수: {prompt_tokens}")
                    except Exception as enc_e: logger.error(f"Tiktoken 인코딩 오류: {enc_e}")
                # LLM 호출
                llm_response_text = call_openai_api(current_recommend_prompt, selected_llm_model, temperature, max_tokens)
                if llm_response_text:
                    recommendations = parse_llm_recommendation_text(llm_response_text)
                    st.session_state.recommendations_faq = recommendations if isinstance(recommendations, list) else []
                    if not st.session_state.recommendations_faq: logger.error("FAQ LLM 응답 파싱 실패."); st.error("FAQ LLM 응답 파싱 실패.")
                else: st.session_state.recommendations_faq = []; logger.error("FAQ LLM 결과 수신 실패."); st.error("FAQ LLM 결과 수신 실패.")
        else: st.session_state.recommendations_faq = []

    st.session_state.processing_done = True

# --- 결과 표시 (세션 상태 기반, 좌우 분리) ---
if st.session_state.user_query and st.session_state.processing_done:
    st.markdown("---")
    st.subheader("1. 쿼리 향상 (LLM)")
    st.text("원래 고객 문의:")
    st.code(st.session_state.user_query, language=None)
    st.text("향상된 검색 쿼리:")
    st.code(st.session_state.enhanced_query, language=None)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # --- 컬럼 1: 상담사 템플릿 추천 ---
    with col1:
        st.header("🧑‍💼 상담사 템플릿 추천")
        st.subheader(f"2-A. 유사도 검색 (상위 {len(st.session_state.candidates_agent)}개)")
        if st.session_state.candidates_agent:
            # retrieve_candidates 반환값의 'TEMPLATE_NAME' 사용
            agent_candidate_options = { f"{i+1}. {c.get('TEMPLATE_NAME', '이름 없음')} (ID: {c['entity_id']}, Score: {c['score']:.4f})": c for i, c in enumerate(st.session_state.candidates_agent) }
            agent_candidate_keys = list(agent_candidate_options.keys())
            def update_selected_agent_candidate(): st.session_state.selected_candidate_agent_key = st.session_state.agent_candidate_selector
            current_agent_key = st.session_state.get("selected_candidate_agent_key")
            default_agent_index = 0
            if current_agent_key in agent_candidate_keys: default_agent_index = agent_candidate_keys.index(current_agent_key)
            elif agent_candidate_keys: st.session_state.selected_candidate_agent_key = agent_candidate_keys[0]
            selected_agent_key = st.selectbox( "검색된 후보 목록 (상담사):", options=agent_candidate_keys, key="agent_candidate_selector", index=default_agent_index, on_change=update_selected_agent_candidate )
            key_to_display_agent = st.session_state.get("selected_candidate_agent_key", selected_agent_key)
            if key_to_display_agent and key_to_display_agent in agent_candidate_options:
                selected_agent_info = agent_candidate_options[key_to_display_agent]
                # retrieve_candidates 반환값의 'CONTENT' 사용
                retrieved_agent_content = selected_agent_info.get('CONTENT', "내용 없음")
                with st.expander("선택된 후보 내용 보기 (상담사)", expanded=False):
                    st.markdown(f"**후보 내용 (ID: {selected_agent_info['entity_id']})**")
                    st.markdown(f"```\n{str(retrieved_agent_content)}\n```")
            elif key_to_display_agent: st.warning(f"선택된 상담사 후보 정보({key_to_display_agent})를 현재 목록에서 찾을 수 없습니다.")
        else: st.info("유사한 상담사 템플릿 후보를 찾지 못했습니다.")

        st.subheader(f"3-A. 최종 추천 (LLM, 상위 {num_recommendations}개)")
        if not st.session_state.recommendations_agent:
             st.error("LLM으로부터 상담사 템플릿 추천 결과를 받지 못했습니다.")
        else:
            top_rec_agent = st.session_state.recommendations_agent[0]
            st.markdown(f"**🥇 최우선 추천 (Rank: {top_rec_agent.get('rank', 1)})**")
            top_rec_agent_id = top_rec_agent.get('entity_id'); top_rec_agent_name = top_rec_agent.get('template_name', '이름 없음')
            st.markdown(f"**ID:** {top_rec_agent_id}"); st.markdown(f"**이름:** {top_rec_agent_name}")
            top_agent_content = "내용 조회 불가"
            if top_rec_agent_id is not None:
                try:
                    content_series = agent_template_df.loc[agent_template_df['ENTITY_ID'] == top_rec_agent_id, 'CONTENT'] # 원본 CONTENT
                    if not content_series.empty and pd.notna(content_series.iloc[0]): top_agent_content = str(content_series.iloc[0])
                    else: logger.warning(f"상담사 ID={top_rec_agent_id} 내용 조회 실패."); top_agent_content = f"오류: ID {top_rec_agent_id} 내용 없음."
                except Exception as e: logger.error(f"상담사 ID={top_rec_agent_id} 내용 조회 오류: {e}", exc_info=True); top_agent_content = f"오류: ID {top_rec_agent_id} 내용 조회 중 문제 발생."
            st.markdown(f"**본문 내용 (ID: {top_rec_agent_id})**"); st.markdown(f"```\n{top_agent_content}\n```")
            if len(st.session_state.recommendations_agent) > 1:
                st.markdown("---"); st.subheader("🥈🥉 추가 추천 (상담사)")
                for i, rec in enumerate(st.session_state.recommendations_agent[1:num_recommendations], start=2):
                     rec_id = rec.get('entity_id'); rec_name = rec.get('template_name', '이름 없음'); rec_rank = rec.get('rank', i)
                     expander_title = f"추천 {rec_rank}: {rec_name} (ID: {rec_id})"
                     with st.expander(expander_title):
                         rec_content = f"오류: ID {rec_id} 내용 조회 중 문제 발생."
                         if rec_id is not None:
                             try:
                                 content_series = agent_template_df.loc[agent_template_df['ENTITY_ID'] == rec_id, 'CONTENT']
                                 if not content_series.empty and pd.notna(content_series.iloc[0]): rec_content = str(content_series.iloc[0])
                                 else: logger.warning(f"상담사 ID={rec_id} 내용 조회 실패."); rec_content = f"오류: ID {rec_id} 내용 없음."
                             except Exception as e: logger.error(f"상담사 ID={rec_id} 내용 조회 오류: {e}", exc_info=True)
                         st.markdown(f"**추가 추천 내용 (ID: {rec_id})**"); st.markdown(f"```\n{rec_content}\n```")

    # --- 컬럼 2: FAQ 추천 ---
    with col2:
        st.header("❓ FAQ 추천")
        st.subheader(f"2-B. 유사도 검색 (상위 {len(st.session_state.candidates_faq)}개)")
        if st.session_state.candidates_faq:
            # retrieve_candidates 반환값의 'TITLE' 사용
            faq_candidate_options = { f"{i+1}. {c.get('TITLE', '제목 없음')} (ID: {c['entity_id']}, Score: {c['score']:.4f})": c for i, c in enumerate(st.session_state.candidates_faq) }
            faq_candidate_keys = list(faq_candidate_options.keys())
            def update_selected_faq_candidate(): st.session_state.selected_candidate_faq_key = st.session_state.faq_candidate_selector
            current_faq_key = st.session_state.get("selected_candidate_faq_key")
            default_faq_index = 0
            if current_faq_key in faq_candidate_keys: default_faq_index = faq_candidate_keys.index(current_faq_key)
            elif faq_candidate_keys: st.session_state.selected_candidate_faq_key = faq_candidate_keys[0]
            selected_faq_key = st.selectbox( "검색된 후보 목록 (FAQ):", options=faq_candidate_keys, key="faq_candidate_selector", index=default_faq_index, on_change=update_selected_faq_candidate )
            key_to_display_faq = st.session_state.get("selected_candidate_faq_key", selected_faq_key)
            if key_to_display_faq and key_to_display_faq in faq_candidate_options:
                selected_faq_info = faq_candidate_options[key_to_display_faq]
                # retrieve_candidates 반환값의 'DESCRIPTION' 사용
                retrieved_faq_content = selected_faq_info.get('DESCRIPTION', "내용 없음")
                with st.expander("선택된 후보 내용 보기 (FAQ)", expanded=False):
                    st.markdown(f"**후보 내용 (ID: {selected_faq_info['entity_id']})**")
                    st.markdown(f"```\n{str(retrieved_faq_content)}\n```")
            elif key_to_display_faq: st.warning(f"선택된 FAQ 후보 정보({key_to_display_faq})를 현재 목록에서 찾을 수 없습니다.")
        else: st.info("유사한 FAQ 후보를 찾지 못했습니다.")

        st.subheader(f"3-B. 최종 추천 (LLM, 상위 {num_recommendations}개)")
        if not st.session_state.recommendations_faq:
             st.error("LLM으로부터 FAQ 추천 결과를 받지 못했습니다.")
        else:
            top_rec_faq = st.session_state.recommendations_faq[0]
            st.markdown(f"**🥇 최우선 추천 (Rank: {top_rec_faq.get('rank', 1)})**")
            top_rec_faq_id = top_rec_faq.get('entity_id')
            # LLM 파싱 결과('template_name')를 제목으로 사용
            top_rec_faq_name = top_rec_faq.get('template_name', '제목 없음')
            st.markdown(f"**ID:** {top_rec_faq_id}")
            st.markdown(f"**제목:** {top_rec_faq_name}")
            top_faq_content = "FAQ 내용을 불러올 수 없습니다."
            if top_rec_faq_id is not None:
                try:
                    # FAQ DataFrame에서 원본 'DESCRIPTION' 조회
                    content_series = faq_template_df.loc[faq_template_df['ENTITY_ID'] == top_rec_faq_id, 'DESCRIPTION']
                    if not content_series.empty and pd.notna(content_series.iloc[0]): top_faq_content = str(content_series.iloc[0])
                    else: logger.warning(f"FAQ ID={top_rec_faq_id} 내용 조회 실패."); top_faq_content = f"오류: ID {top_rec_faq_id} 내용 없음."
                except Exception as e: logger.error(f"FAQ ID={top_rec_faq_id} 내용 조회 오류: {e}", exc_info=True); top_faq_content = f"오류: ID {top_rec_faq_id} 내용 조회 중 문제 발생."
            st.markdown(f"**내용 (ID: {top_rec_faq_id})**")
            st.markdown(f"```\n{top_faq_content}\n```")
            if len(st.session_state.recommendations_faq) > 1:
                st.markdown("---"); st.subheader("🥈🥉 추가 추천 (FAQ)")
                for i, rec in enumerate(st.session_state.recommendations_faq[1:num_recommendations], start=2):
                     rec_id = rec.get('entity_id'); rec_name = rec.get('template_name', '제목 없음'); rec_rank = rec.get('rank', i)
                     expander_title = f"추천 {rec_rank}: {rec_name} (ID: {rec_id})"
                     with st.expander(expander_title):
                         rec_content = f"오류: ID {rec_id} 내용 조회 중 문제 발생."
                         if rec_id is not None:
                             try:
                                 content_series = faq_template_df.loc[faq_template_df['ENTITY_ID'] == rec_id, 'DESCRIPTION']
                                 if not content_series.empty and pd.notna(content_series.iloc[0]): rec_content = str(content_series.iloc[0])
                                 else: logger.warning(f"FAQ ID={rec_id} 내용 조회 실패."); rec_content = f"오류: ID {rec_id} 내용 없음."
                             except Exception as e: logger.error(f"FAQ ID={rec_id} 내용 조회 오류: {e}", exc_info=True)
                         st.markdown(f"**추가 추천 내용 (ID: {rec_id})**")
                         st.markdown(f"```\n{rec_content}\n```")

# 초기 안내 메시지
elif not st.session_state.user_query:
    st.info("하단 입력창에 고객 문의 내용을 입력하고 Enter 키를 누르거나 보내기 버튼을 클릭하세요.")

logger.info("Streamlit 앱 렌더링 완료.")