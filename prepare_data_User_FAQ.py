# prepare_data_User_FAQ.py

import pandas as pd
import numpy as np
import re
import os
import time
import faiss
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import io # StringIO는 사용하지 않지만, 혹시 모를 대비로 남겨둠

# --- 로깅 설정 ---
log_dir = 'log'
if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
# 로그 파일명에 User_FAQ 명시
log_filename = os.path.join(log_dir, f"prepare_data_User_FAQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = logging.getLogger('PrepareUserFAQLogger') # 로거 이름 변경
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(log_filename, encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)
logger.info("="*20 + " 사용자 FAQ 데이터 준비 시작 " + "="*20)
# --- 로깅 설정 끝 ---

# --- 설정값 ---
FAQ_TEMPLATE_FILEPATH = 'User_FAQ_templates.txt' # 입력 파일 경로 변경
EMBEDDING_MODEL_NAME = 'jhgan/ko-sbert-sts' # 동일 모델 사용 또는 변경 가능
# 출력 파일명 변경
FAQ_EMBEDDING_FILENAME = 'faq_embeddings.npy'
FAQ_IDS_FILENAME = 'faq_ids.npy'
FAQ_INDEX_FILENAME = 'faq_faiss_index.index'
FAQ_DEBUG_CSV_FILENAME = 'debug_faq_template_data.csv'
# --- 설정값 끝 ---

def load_and_clean_faq_templates(filepath):
    """사용자 FAQ 템플릿 파일을 파싱하여 DataFrame으로 로드하고 정제합니다."""
    logger.info(f"FAQ 템플릿 데이터 로드 시도: '{filepath}'")
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
                        # CLEANED_DESCRIPTION 생성 (줄바꿈, 탭 등을 공백으로)
                        cleaned_desc = re.sub(r'[\n\r\t]+', ' ', current_faq['DESCRIPTION'])
                        cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc).strip()
                        current_faq['CLEANED_DESCRIPTION'] = cleaned_desc

                        current_faq.setdefault('ENTITY_ID', None)
                        current_faq.setdefault('CATEGORY_DETAIL_ENTITY_ID', None) # 추가된 필드
                        current_faq.setdefault('TITLE', '제목 없음')

                        if current_faq.get('ENTITY_ID') is not None and current_faq.get('CLEANED_DESCRIPTION'):
                            try:
                                current_faq['ENTITY_ID'] = int(current_faq['ENTITY_ID'])
                                # CATEGORY_DETAIL_ENTITY_ID도 숫자로 변환 시도
                                cat_id = current_faq.get('CATEGORY_DETAIL_ENTITY_ID')
                                current_faq['CATEGORY_DETAIL_ENTITY_ID'] = int(cat_id) if cat_id else None
                                faq_data.append(current_faq.copy())
                            except (ValueError, TypeError):
                                logger.warning(f"잘못된 ID 값 (라인 ~{line_num}): {current_faq.get('ENTITY_ID')}, {current_faq.get('CATEGORY_DETAIL_ENTITY_ID')}")
                        else:
                            logger.warning(f"FAQ 섹션 데이터 누락 또는 내용 없음 (라인 ~{line_num}): {current_faq.get('ENTITY_ID')}")

                    current_faq = {}; description_lines = []; is_description_section = False
                elif ':' in stripped_line and not is_description_section:
                    try:
                        key, value = stripped_line.split(':', 1)
                        key = key.strip(); value = value.strip()
                        if key in ['ENTITY_ID', 'CATEGORY_DETAIL_ENTITY_ID', 'TITLE']:
                            current_faq[key] = value
                        elif key == 'DESCRIPTION':
                            is_description_section = True
                            if value: description_lines.append(value)
                        else: logger.warning(f"예상치 못한 키 발견 (라인 {line_num}): {key}")
                    except ValueError: logger.warning(f"키:값 파싱 오류 (라인 {line_num}): {stripped_line}")
                else:
                    if not is_description_section and not line.strip(): continue
                    is_description_section = True
                    description_lines.append(line.rstrip()) # 원본 줄바꿈 유지 위해 rstrip()만 사용

        # 마지막 섹션 처리
        if current_faq:
            current_faq['DESCRIPTION'] = "\n".join(description_lines).strip()
            cleaned_desc = re.sub(r'[\n\r\t]+', ' ', current_faq['DESCRIPTION'])
            cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc).strip()
            current_faq['CLEANED_DESCRIPTION'] = cleaned_desc
            current_faq.setdefault('ENTITY_ID', None); current_faq.setdefault('CATEGORY_DETAIL_ENTITY_ID', None); current_faq.setdefault('TITLE', '제목 없음')
            if current_faq.get('ENTITY_ID') is not None and current_faq.get('CLEANED_DESCRIPTION'):
                 try:
                     current_faq['ENTITY_ID'] = int(current_faq['ENTITY_ID'])
                     cat_id = current_faq.get('CATEGORY_DETAIL_ENTITY_ID')
                     current_faq['CATEGORY_DETAIL_ENTITY_ID'] = int(cat_id) if cat_id else None
                     faq_data.append(current_faq.copy())
                 except (ValueError, TypeError): logger.warning(f"잘못된 ID 값 (마지막 섹션): {current_faq.get('ENTITY_ID')}, {current_faq.get('CATEGORY_DETAIL_ENTITY_ID')}")
            else: logger.warning(f"마지막 섹션 데이터 누락 또는 내용 없음: {current_faq.get('ENTITY_ID')}")

        if not faq_data: logger.error("파일에서 유효한 FAQ 데이터를 파싱하지 못했습니다."); return None

        df = pd.DataFrame(faq_data)

        # 데이터 타입 최종 확인 및 NaN 처리
        df['ENTITY_ID'] = df['ENTITY_ID'].astype(int)
        # CATEGORY_DETAIL_ENTITY_ID는 None일 수 있으므로 object 타입 유지 또는 Int64 사용
        df['CATEGORY_DETAIL_ENTITY_ID'] = pd.to_numeric(df['CATEGORY_DETAIL_ENTITY_ID'], errors='coerce').astype('Int64') # Nullable Integer
        for col in ['TITLE', 'DESCRIPTION', 'CLEANED_DESCRIPTION']:
             if col in df.columns: df[col] = df[col].fillna('').astype(str)
             else: df[col] = ''; logger.warning(f"'{col}' 컬럼이 생성되지 않아 빈 컬럼으로 추가합니다.")
        df = df[df['CLEANED_DESCRIPTION'].str.strip() != ''] # 내용 없는 행 제거

        logger.info(f"총 {len(df)}개의 유효 FAQ 로드 및 정제 완료.")
        return df

    except Exception as e:
        logger.error(f"FAQ 파일 처리 중 오류: {e}", exc_info=True)
        return None

def generate_and_save_faq_embeddings(df, model_name, embedding_filename, ids_filename):
    """FAQ 임베딩 생성 및 저장 (TITLE + DESCRIPTION 기반)"""
    logger.info(f"임베딩 모델 로드 시도: '{model_name}'")
    try: model = SentenceTransformer(model_name); logger.info(f"임베딩 모델 '{model_name}' 로드 완료.")
    except Exception as e: logger.error(f"임베딩 모델 로드 오류: {e}", exc_info=True); return None, None

    logger.info("\nFAQ 임베딩 생성 시작 (제목 + 내용 기반)...")
    # 임베딩 대상: TITLE과 CLEANED_DESCRIPTION 조합
    texts_to_embed = (df['TITLE'].astype(str).fillna('') + " :: " + df['CLEANED_DESCRIPTION'].astype(str).fillna('')).tolist()
    logger.info(f"임베딩 대상 텍스트 예시 (첫 번째): {texts_to_embed[0][:100]}...")

    start_time = time.time()
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    end_time = time.time()
    logger.info(f"FAQ 임베딩 생성 완료! 형태: {embeddings.shape}, 소요 시간: {end_time - start_time:.2f} 초")

    try:
        np.save(embedding_filename, embeddings); logger.info(f"FAQ 임베딩을 '{embedding_filename}' 파일로 저장.")
        ids = df['ENTITY_ID'].values; np.save(ids_filename, ids); logger.info(f"FAQ ID를 '{ids_filename}' 파일로 저장.")
        return embeddings, ids
    except Exception as e: logger.error(f"FAQ 임베딩/ID 저장 오류: {e}", exc_info=True); return None, None

def build_and_save_faq_faiss_index(embeddings, index_filename):
    """FAQ Faiss 인덱스 구축 및 저장"""
    if embeddings is None: logger.error("FAQ 임베딩 데이터가 없어 Faiss 인덱스 구축 불가."); return None
    logger.info("\nFAQ Faiss 인덱스 구축 시작...")
    dimension = embeddings.shape[1]; logger.info(f"벡터 차원: {dimension}")
    index = faiss.IndexFlatIP(dimension); logger.info("IndexFlatIP 인덱스 생성 완료.")
    logger.info(f"{embeddings.shape[0]}개의 벡터를 인덱스에 추가합니다...")
    start_time = time.time(); index.add(embeddings.astype('float32')); end_time = time.time()
    logger.info(f"벡터 추가 완료. 소요 시간: {end_time - start_time:.2f} 초"); logger.info(f"인덱스 내 총 벡터 수: {index.ntotal}")
    try: faiss.write_index(index, index_filename); logger.info(f"FAQ Faiss 인덱스를 '{index_filename}' 파일로 저장."); return index
    except Exception as e: logger.error(f"FAQ Faiss 인덱스 저장 오류: {e}", exc_info=True); return None

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 데이터 로드 및 정제
    faq_df = load_and_clean_faq_templates(FAQ_TEMPLATE_FILEPATH)

    if faq_df is not None:
        # --- !!! CSV 파일 저장 (검증용) !!! ---
        try:
            logger.info(f"저장될 FAQ DataFrame 정보:\n{faq_df.info()}")
            logger.debug(f"저장될 FAQ DataFrame 샘플:\n{faq_df.head().to_string()}")
            faq_df.to_csv(FAQ_DEBUG_CSV_FILENAME, index=False, encoding='utf-8-sig')
            logger.info(f"FAQ DataFrame 데이터를 '{FAQ_DEBUG_CSV_FILENAME}'으로 저장했습니다.")
        except Exception as e: logger.error(f"FAQ CSV 파일 저장 오류: {e}", exc_info=True)
        # --- !!! CSV 저장 끝 !!! ---

        # 2. 임베딩 생성 및 저장
        faq_embeddings, faq_ids = generate_and_save_faq_embeddings(
            faq_df, EMBEDDING_MODEL_NAME, FAQ_EMBEDDING_FILENAME, FAQ_IDS_FILENAME
        )

        if faq_embeddings is not None:
            # 3. Faiss 인덱스 구축 및 저장
            build_and_save_faq_faiss_index(faq_embeddings, FAQ_INDEX_FILENAME)

    logger.info("="*20 + " 사용자 FAQ 데이터 준비 완료 " + "="*20)
# --- 메인 실행 로직 끝 ---