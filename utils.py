# utils.py (최종 버전)

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import os
import time
import openai
import json
import logging

# --- 로거 설정 ---
# app.py에서 설정된 로거를 사용합니다.
logger = logging.getLogger('StreamlitAppLogger')
# --- 로거 설정 끝 ---


# --- 리소스 로드 함수 ---
def load_embedding_model(model_name):
    """임베딩 모델을 로드합니다."""
    logger.info(f"임베딩 모델 로드 시도: '{model_name}'")
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"임베딩 모델 '{model_name}' 로드 완료.")
        return model
    except Exception as e:
        logger.error(f"임베딩 모델 로드 오류: {e}", exc_info=True)
        return None

def load_faiss_index_and_ids(index_path, ids_path):
    """저장된 Faiss 인덱스와 ID 배열을 로드합니다."""
    logger.info(f"Faiss 인덱스 및 ID 로드 시도: '{index_path}', '{ids_path}'")
    if not os.path.exists(index_path) or not os.path.exists(ids_path):
        logger.error(f"파일 부재 오류: '{index_path}' 또는 '{ids_path}'")
        return None, None
    try:
        index = faiss.read_index(index_path)
        ids = np.load(ids_path)
        logger.info(f"Faiss 인덱스 로드 완료 (총 {index.ntotal}개 벡터).")
        logger.info(f"템플릿/FAQ ID 로드 완료 (총 {len(ids)}개 ID).") # 로그 메시지 일반화
        return index, ids
    except Exception as e:
        logger.error(f"인덱스/ID 파일 로드 오류: {e}", exc_info=True)
        return None, None

# 상담사 템플릿 로드 함수
def load_template_dataframe(filepath):
    """상담사 템플릿 파일을 파싱하여 DataFrame으로 로드하고 정제합니다."""
    logger.info(f"상담사 템플릿 데이터 로드 시도: '{filepath}'")
    if not os.path.exists(filepath): logger.error(f"파일 부재 오류: {filepath}"); return None

    templates_data = []
    current_template = {}
    content_lines = []
    is_content_section = False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                processed_line = line.rstrip()
                if processed_line.strip() == '---':
                    if current_template:
                        current_template['CONTENT'] = "\n".join(content_lines).strip()
                        cleaned_content = re.sub(r'[\n\r\t]+', ' ', current_template['CONTENT'])
                        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
                        current_template['CLEANED_CONTENT'] = cleaned_content
                        current_template.setdefault('ENTITY_ID', None); current_template.setdefault('COMPANY_CODE', ''); current_template.setdefault('TEMPLATE_NAME', '이름 없음')
                        if current_template.get('ENTITY_ID') is not None and current_template.get('CLEANED_CONTENT'):
                             try:
                                 current_template['ENTITY_ID'] = int(current_template['ENTITY_ID'])
                                 templates_data.append(current_template.copy())
                             except (ValueError, TypeError): logger.warning(f"상담사 잘못된 ENTITY_ID 값 (라인 ~{line_num}): {current_template.get('ENTITY_ID')}")
                        else: logger.warning(f"상담사 섹션 데이터 누락 또는 내용 없음 (라인 ~{line_num}): {current_template.get('ENTITY_ID')}")
                    current_template = {}; content_lines = []; is_content_section = False
                elif ':' in line.strip() and not is_content_section:
                    try:
                        key, value = line.strip().split(':', 1); key = key.strip(); value = value.strip()
                        if key in ['ENTITY_ID', 'COMPANY_CODE', 'TEMPLATE_NAME']: current_template[key] = value
                        elif key == 'CONTENT': is_content_section = True; content_lines.append(value) if value else None
                        else: logger.warning(f"상담사 예상치 못한 키 발견 (라인 {line_num}): {key}")
                    except ValueError: logger.warning(f"상담사 키:값 파싱 오류 (라인 {line_num}): {line.strip()}")
                else:
                    if not is_content_section and not line.strip(): continue
                    is_content_section = True; content_lines.append(line.rstrip())
        if current_template:
            current_template['CONTENT'] = "\n".join(content_lines).strip()
            cleaned_content = re.sub(r'[\n\r\t]+', ' ', current_template['CONTENT']); cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
            current_template['CLEANED_CONTENT'] = cleaned_content
            current_template.setdefault('ENTITY_ID', None); current_template.setdefault('COMPANY_CODE', ''); current_template.setdefault('TEMPLATE_NAME', '이름 없음')
            if current_template.get('ENTITY_ID') is not None and current_template.get('CLEANED_CONTENT'):
                 try: current_template['ENTITY_ID'] = int(current_template['ENTITY_ID']); templates_data.append(current_template.copy())
                 except (ValueError, TypeError): logger.warning(f"상담사 잘못된 ENTITY_ID 값 (마지막 섹션): {current_template.get('ENTITY_ID')}")
            else: logger.warning(f"상담사 마지막 섹션 데이터 누락 또는 내용 없음: {current_template.get('ENTITY_ID')}")
    except Exception as e: logger.error(f"상담사 템플릿 파일 읽기 오류: {e}", exc_info=True); return None
    if not templates_data: logger.error("파일에서 유효한 상담사 템플릿 데이터를 파싱하지 못했습니다."); return None
    df = pd.DataFrame(templates_data)
    df['ENTITY_ID'] = df['ENTITY_ID'].astype(int)
    for col in ['COMPANY_CODE', 'TEMPLATE_NAME', 'CONTENT', 'CLEANED_CONTENT']:
         if col in df.columns: df[col] = df[col].fillna('').astype(str)
         else: df[col] = ''; logger.warning(f"상담사 '{col}' 컬럼이 생성되지 않아 빈 컬럼으로 추가합니다.")
    df = df[df['CLEANED_CONTENT'].str.strip() != '']
    logger.info(f"총 {len(df)}개의 유효 상담사 템플릿 로드 및 정제 완료.")
    return df

# FAQ 데이터 로드 함수
def load_faq_dataframe(filepath):
    """사용자 FAQ 템플릿 파일을 파싱하여 DataFrame으로 로드하고 정제합니다."""
    logger.info(f"FAQ 데이터 로드 시도: '{filepath}'")
    if not os.path.exists(filepath): logger.error(f"파일 부재 오류: {filepath}"); return None

    faq_data = []
    current_faq = {}
    description_lines = []
    is_description_section = False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()
                if stripped_line == '---':
                    if current_faq:
                        current_faq['DESCRIPTION'] = "\n".join(description_lines).strip()
                        cleaned_desc = re.sub(r'[\n\r\t]+', ' ', current_faq['DESCRIPTION'])
                        cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc).strip()
                        current_faq['CLEANED_DESCRIPTION'] = cleaned_desc
                        current_faq.setdefault('ENTITY_ID', None); current_faq.setdefault('CATEGORY_DETAIL_ENTITY_ID', None); current_faq.setdefault('TITLE', '제목 없음')
                        if current_faq.get('ENTITY_ID') is not None and current_faq.get('CLEANED_DESCRIPTION'):
                            try:
                                current_faq['ENTITY_ID'] = int(current_faq['ENTITY_ID'])
                                cat_id = current_faq.get('CATEGORY_DETAIL_ENTITY_ID'); current_faq['CATEGORY_DETAIL_ENTITY_ID'] = int(cat_id) if cat_id else None
                                faq_data.append(current_faq.copy())
                            except (ValueError, TypeError): logger.warning(f"FAQ 잘못된 ID 값 (라인 ~{line_num}): {current_faq.get('ENTITY_ID')}, {current_faq.get('CATEGORY_DETAIL_ENTITY_ID')}")
                        else: logger.warning(f"FAQ 섹션 데이터 누락 또는 내용 없음 (라인 ~{line_num}): {current_faq.get('ENTITY_ID')}")
                    current_faq = {}; description_lines = []; is_description_section = False
                elif ':' in stripped_line and not is_description_section:
                    try:
                        key, value = stripped_line.split(':', 1); key = key.strip(); value = value.strip()
                        if key in ['ENTITY_ID', 'CATEGORY_DETAIL_ENTITY_ID', 'TITLE']: current_faq[key] = value
                        elif key == 'DESCRIPTION': is_description_section = True; description_lines.append(value) if value else None
                        else: logger.warning(f"FAQ 예상치 못한 키 발견 (라인 {line_num}): {key}")
                    except ValueError: logger.warning(f"FAQ 키:값 파싱 오류 (라인 {line_num}): {stripped_line}")
                else:
                    if not is_description_section and not line.strip(): continue
                    is_description_section = True; description_lines.append(line.rstrip())
        if current_faq:
            current_faq['DESCRIPTION'] = "\n".join(description_lines).strip()
            cleaned_desc = re.sub(r'[\n\r\t]+', ' ', current_faq['DESCRIPTION']); cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc).strip()
            current_faq['CLEANED_DESCRIPTION'] = cleaned_desc
            current_faq.setdefault('ENTITY_ID', None); current_faq.setdefault('CATEGORY_DETAIL_ENTITY_ID', None); current_faq.setdefault('TITLE', '제목 없음')
            if current_faq.get('ENTITY_ID') is not None and current_faq.get('CLEANED_DESCRIPTION'):
                 try:
                     current_faq['ENTITY_ID'] = int(current_faq['ENTITY_ID']); cat_id = current_faq.get('CATEGORY_DETAIL_ENTITY_ID'); current_faq['CATEGORY_DETAIL_ENTITY_ID'] = int(cat_id) if cat_id else None
                     faq_data.append(current_faq.copy())
                 except (ValueError, TypeError): logger.warning(f"FAQ 잘못된 ID 값 (마지막 섹션): {current_faq.get('ENTITY_ID')}, {current_faq.get('CATEGORY_DETAIL_ENTITY_ID')}")
            else: logger.warning(f"FAQ 마지막 섹션 데이터 누락 또는 내용 없음: {current_faq.get('ENTITY_ID')}")
    except Exception as e: logger.error(f"FAQ 파일 읽기 오류: {e}", exc_info=True); return None
    if not faq_data: logger.error("파일에서 유효한 FAQ 데이터를 파싱하지 못했습니다."); return None
    df = pd.DataFrame(faq_data)
    df['ENTITY_ID'] = df['ENTITY_ID'].astype(int)
    df['CATEGORY_DETAIL_ENTITY_ID'] = pd.to_numeric(df['CATEGORY_DETAIL_ENTITY_ID'], errors='coerce').astype('Int64')
    # FAQ 컬럼명 유지: TITLE, DESCRIPTION, CLEANED_DESCRIPTION
    for col in ['TITLE', 'DESCRIPTION', 'CLEANED_DESCRIPTION']:
         if col in df.columns: df[col] = df[col].fillna('').astype(str)
         else: df[col] = ''; logger.warning(f"FAQ '{col}' 컬럼이 생성되지 않아 빈 컬럼으로 추가합니다.")
    df = df[df['CLEANED_DESCRIPTION'].str.strip() != '']
    logger.info(f"총 {len(df)}개의 유효 FAQ 로드 및 정제 완료.")
    return df


# --- LLM 관련 함수 ---
def call_openai_api(prompt, model, temperature, max_tokens):
    """OpenAI API를 호출하고 일반 텍스트 응답을 반환합니다."""
    logger.info(f"OpenAI API 호출 시작 (텍스트 응답, 모델: {model}, 온도: {temperature}, 최대 토큰: {max_tokens})")
    try:
        start_time = time.time()
        response = openai.chat.completions.create(
            model=model, messages=[ {"role": "system", "content": "당신은 지시에 따라 정확하고 간결하게 답변하는 AI입니다."}, {"role": "user", "content": prompt} ],
            temperature=temperature, max_tokens=max_tokens
        )
        end_time = time.time()
        output_text = response.choices[0].message.content
        logger.info(f"OpenAI API 응답 수신 완료 ({end_time - start_time:.2f} 초)")
        logger.debug(f"LLM 원본 응답 (텍스트): {output_text}")
        # 디버깅 파일 저장
        debug_print_to_file("LLM Response (Text)", output_text, "debug_response.txt")
        return output_text.strip() if output_text else None
    except Exception as e: logger.error(f"OpenAI API 호출 오류 (텍스트): {e}", exc_info=True); return None

# --- LLM 텍스트 응답 파싱 함수 ---
def parse_llm_recommendation_text(text_response):
    """LLM 텍스트 응답을 파싱하여 추천 목록(딕셔너리 리스트)으로 변환합니다."""
    recommendations = []
    if not text_response: logger.warning("LLM 응답 텍스트가 비어있어 파싱 불가."); return recommendations
    lines = text_response.strip().split('\n')
    logger.info(f"LLM 응답 파싱 시도 (총 {len(lines)}줄)")
    logger.debug(f"파싱 대상 텍스트:\n{text_response}")
    pattern = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*:\s*(.+)$")
    for i, line in enumerate(lines):
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            try:
                rank = int(match.group(1)); entity_id = int(match.group(2)); template_name = match.group(3).strip()
                recommendations.append({ "rank": rank, "entity_id": entity_id, "template_name": template_name }) # 파싱 결과는 'template_name' 키 사용
                logger.info(f"파싱 성공 (줄 {i+1}): Rank={rank}, ID={entity_id}, Name='{template_name}'")
            except ValueError: logger.warning(f"파싱 중 값 변환 오류 (줄 {i+1}): '{line}'")
            except Exception as e: logger.error(f"파싱 중 예상치 못한 오류 (줄 {i+1}): '{line}', 오류: {e}", exc_info=True)
        else: logger.warning(f"파싱 실패: 예상된 형식 아님 (줄 {i+1}): '{line}'")
    recommendations.sort(key=lambda x: x.get('rank', float('inf')))
    if not recommendations: logger.error("LLM 응답 텍스트에서 유효한 추천 정보 미추출.")
    return recommendations

# --- 검색 함수 ---
def retrieve_candidates(query, embedding_model, faiss_index, template_ids, template_df, k):
    """쿼리 임베딩 후 Faiss 검색을 수행하여 후보 리스트를 반환합니다."""
    logger.info(f"후보 검색 시작 - 쿼리: \"{query}\", k={k}")
    if embedding_model is None or faiss_index is None or template_ids is None or template_df is None: logger.error("검색 리소스 미준비 오류."); return []
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding.astype('float32'), k)
        candidates = []
        if len(indices) > 0 and len(indices[0]) > 0:
            for i in range(len(indices[0])):
                vector_index = indices[0][i]
                if 0 <= vector_index < len(template_ids):
                    template_entity_id = template_ids[vector_index]
                    try:
                        template_rows = template_df[template_df['ENTITY_ID'] == template_entity_id]
                        if not template_rows.empty:
                            template_row = template_rows.iloc[0]
                            score = distances[0][i] if i < len(distances[0]) else -1.0
                            # 반환 딕셔너리에 필요한 모든 원본 컬럼 포함 (app.py에서 사용하기 위함)
                            candidate_data = {
                                "entity_id": int(template_entity_id),
                                "score": float(score)
                            }
                            # DataFrame에 있는 컬럼만 추가
                            for col in ['TEMPLATE_NAME', 'TITLE', 'CONTENT', 'DESCRIPTION', 'CLEANED_CONTENT', 'CLEANED_DESCRIPTION']:
                                if col in template_row:
                                    candidate_data[col] = str(template_row.get(col, ''))
                            candidates.append(candidate_data)
                        else: logger.warning(f"ID={template_entity_id} 정보 조회 실패: DataFrame에 해당 ID 없음.")
                    except KeyError as ke: logger.warning(f"ID={template_entity_id} 정보 조회 실패 (KeyError: {ke}): 필요한 컬럼 없음.")
                    except Exception as inner_e: logger.error(f"ID={template_entity_id} 정보 조회 중 예상치 못한 오류: {inner_e}", exc_info=True)
                else: logger.warning(f"잘못된 벡터 인덱스({vector_index})가 검색 결과에 포함됨 (총 ID 개수: {len(template_ids)}).")
        else: logger.warning("Faiss 검색 결과 인덱스가 비어 있음.")
        logger.info(f"후보 {len(candidates)}개 검색 완료.")
        return candidates
    except Exception as e: logger.error(f"후보 검색 함수 실행 중 오류 발생: {e}", exc_info=True); return []

# --- 프롬프트 관리 함수 ---
def load_prompt(filepath):
    """파일에서 프롬프트 템플릿을 로드합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: logger.warning(f"프롬프트 파일 '{filepath}'를 찾을 수 없음."); return None
    except Exception as e: logger.error(f"프롬프트 파일 '{filepath}' 로드 중 오류: {e}", exc_info=True); return None

def save_prompt(filepath, content):
    """프롬프트 템플릿을 파일에 저장합니다."""
    try:
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path): os.makedirs(dir_path); logger.info(f"디렉토리 생성: '{dir_path}'")
        with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
        logger.info(f"프롬프트를 '{filepath}'에 저장했습니다."); return True
    except Exception as e: logger.error(f"프롬프트 파일 '{filepath}' 저장 중 오류: {e}", exc_info=True); return False

# --- 디버깅 파일 출력 함수 ---
def debug_print_to_file(title, content, filename):
    """디버그 내용을 파일에 기록합니다."""
    filepath = os.path.join('debug', filename) # debug 디렉토리 사용
    try:
        os.makedirs('debug', exist_ok=True) # 디렉토리 없으면 생성
        # 'a' 모드로 열어 기존 내용에 추가, utf-8 인코딩 명시
        with open(filepath, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n\n{'='*30} {timestamp} {'='*30}\n")
            f.write(f"--- {title} ---\n")
            f.write(str(content))
            f.write(f"\n{'='*60}\n")
        logger.info(f"디버그 정보 저장 완료: {filepath}")
    except Exception as e:
        logger.error(f"디버그 파일 저장 오류 ({filepath}): {e}", exc_info=True)