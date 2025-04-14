# prepare_data.py (전체 코드 - 파싱 로직 개선 및 로깅 복구 버전)

import pandas as pd
import numpy as np
import re
import os
import time
import faiss
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import io # StringIO 사용 위해 추가

# --- 로깅 설정 ---
log_dir = 'log'
if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"prepare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = logging.getLogger('PrepareDataLogger')
logger.setLevel(logging.INFO) # 기본 INFO 레벨
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 파일 핸들러
fh = logging.FileHandler(log_filename, encoding='utf-8')
fh.setLevel(logging.INFO) # 파일에도 INFO 레벨
fh.setFormatter(formatter)
# 콘솔 핸들러
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) # 콘솔에도 INFO 레벨
ch.setFormatter(formatter)
# 핸들러 중복 추가 방지
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)
logger.info("="*20 + " 데이터 준비 시작 " + "="*20)
# --- 로깅 설정 끝 ---

# --- 설정값 ---
TEMPLATE_FILEPATH = 'structured_templates_250412_1.txt'
EMBEDDING_MODEL_NAME = 'jhgan/ko-sbert-sts' # 또는 다른 모델
EMBEDDING_FILENAME = 'template_embeddings.npy'
IDS_FILENAME = 'template_ids.npy'
INDEX_FILENAME = 'faiss_index.index'
DEBUG_CSV_FILENAME = 'debug_template_data_final_v2.csv' # 새 CSV 파일명
# --- 설정값 끝 ---

# prepare_data.py의 load_and_clean_templates 함수에서 정제 로직 수정
def load_and_clean_templates(filepath):
    """템플릿 텍스트 파일을 직접 파싱하여 DataFrame으로 로드하고 정제합니다."""
    logger.info(f"템플릿 데이터 로드 시도 (직접 파싱): '{filepath}'")
    if not os.path.exists(filepath): logger.error(f"파일 부재 오류: {filepath}"); return None

    templates_data = []
    current_template = {}
    content_lines = [] # CONTENT 라인을 모으는 리스트
    is_content_section = False # 현재 CONTENT 섹션인지 여부 플래그

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 원본 라인의 앞뒤 공백만 제거 (중간 공백, 줄바꿈 유지 위함)
                processed_line = line.rstrip() # 오른쪽 공백/줄바꿈만 제거하거나, 필요시 line 그대로 사용

                if processed_line.strip() == '---': # 섹션 구분자 (빈 줄 고려)
                    if current_template: # 이전 섹션 데이터 처리
                        # content_lines에 모인 내용을 합쳐 CONTENT 필드 생성
                        # join 시 각 라인의 원본 줄바꿈이 유지됨
                        current_template['CONTENT'] = "\n".join(content_lines).strip()
                        # CLEANED_CONTENT 생성 (여기서 줄바꿈, 탭 등을 공백으로 변경)
                        # cleaned_content = re.sub(r'[\n\r\t]+', ' ', current_template['CONTENT'])
                        # cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
                        # current_template['CLEANED_CONTENT'] = cleaned_content
                        current_template['CLEANED_CONTENT'] = current_template['CONTENT']  # 원본 그대로 유지

                        # 필수 필드 기본값 설정 및 리스트 추가
                        current_template.setdefault('ENTITY_ID', None)
                        current_template.setdefault('COMPANY_CODE', '')
                        current_template.setdefault('TEMPLATE_NAME', '이름 없음')

                        # ID가 있고 내용이 있는 경우에만 추가
                        if current_template.get('ENTITY_ID') is not None and current_template.get('CLEANED_CONTENT'):
                             # ID를 숫자로 변환 시도
                             try:
                                 current_template['ENTITY_ID'] = int(current_template['ENTITY_ID'])
                                 templates_data.append(current_template.copy()) # copy() 중요!
                             except (ValueError, TypeError):
                                 logger.warning(f"잘못된 ENTITY_ID 값 (라인 ~{line_num}): {current_template.get('ENTITY_ID')}")
                        else:
                             logger.warning(f"섹션 데이터 누락 또는 내용 없음 (라인 ~{line_num}): {current_template.get('ENTITY_ID')}")

                    current_template = {} # 새 섹션 초기화
                    content_lines = [] # 버퍼 초기화
                    is_content_section = False # 플래그 초기화

                elif ':' in line.strip() and not is_content_section: # CONTENT 시작 전의 키:값 라인
                    try:
                        key, value = line.strip().split(':', 1)
                        key = key.strip(); value = value.strip()
                        if key in ['ENTITY_ID', 'COMPANY_CODE', 'TEMPLATE_NAME']:
                            current_template[key] = value
                        elif key == 'CONTENT': # CONTENT 키 발견
                            is_content_section = True
                            if value: content_lines.append(value) # 첫 줄 내용 추가
                        else: # 예상치 못한 키
                             # CONTENT로 처리할 수도 있음 (선택적)
                             # is_content_section = True
                             # content_lines.append(line.strip())
                             logger.warning(f"예상치 못한 키 발견 (라인 {line_num}): {key}")
                    except ValueError:
                         logger.warning(f"키:값 파싱 오류 (라인 {line_num}): {line.strip()}")
                         # 오류 라인도 내용으로 처리?
                         # is_content_section = True
                         # content_lines.append(line.strip())
                else: # CONTENT 섹션이 시작되었거나 ':' 없는 라인
                    # 첫 줄이 비어있고 ':' 없는 라인이면 무시할 수도 있음 (주석 등)
                    if not is_content_section and not line.strip():
                        continue
                    is_content_section = True
                    content_lines.append(line.rstrip()) # 오른쪽 공백/줄바꿈만 제거하고 추가

        # 파일 끝에 도달 후 마지막 섹션 처리
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
                 except (ValueError, TypeError):
                      logger.warning(f"잘못된 ENTITY_ID 값 (마지막 섹션): {current_template.get('ENTITY_ID')}")
            else: logger.warning(f"마지막 섹션 데이터 누락 또는 내용 없음: {current_template.get('ENTITY_ID')}")

        if not templates_data: logger.error("파일에서 유효한 템플릿 데이터를 파싱하지 못했습니다."); return None

        # DataFrame 생성
        df = pd.DataFrame(templates_data)

        # 데이터 타입 최종 확인 및 NaN 처리
        df['ENTITY_ID'] = df['ENTITY_ID'].astype(int) # 이미 int로 변환 시도했으므로 바로 변환
        for col in ['COMPANY_CODE', 'TEMPLATE_NAME', 'CONTENT', 'CLEANED_CONTENT']:
             if col in df.columns: df[col] = df[col].fillna('').astype(str)
             else: df[col] = ''; logger.warning(f"'{col}' 컬럼이 생성되지 않아 빈 컬럼으로 추가합니다.")

        logger.info(f"총 {len(df)}개의 유효 템플릿 로드 및 정제 완료 (직접 파싱 v2).")
        return df

    except Exception as e:
        logger.error(f"템플릿 파일 처리 중 오류: {e}", exc_info=True)
        return None

def generate_and_save_embeddings(df, model_name, embedding_filename, ids_filename):
    """임베딩 생성 및 저장"""
    logger.info(f"임베딩 모델 로드 시도: '{model_name}'")
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"임베딩 모델 '{model_name}' 로드 완료.")
    except Exception as e:
        logger.error(f"임베딩 모델 로드 오류: {e}", exc_info=True)
        return None, None

    logger.info("\n템플릿 임베딩 생성 시작 (제목 + 내용 기반)...")
    # 임베딩 대상 텍스트 생성 시 NaN 처리 강화
    texts_to_embed = (df['TEMPLATE_NAME'].astype(str).fillna('') + " :: " + df['CLEANED_CONTENT'].astype(str).fillna('')).tolist()
    logger.info(f"임베딩 대상 텍스트 예시 (첫 번째): {texts_to_embed[0][:100]}...")

    start_time = time.time()
    embeddings = model.encode(texts_to_embed, show_progress_bar=True) # 진행률 표시 복구
    end_time = time.time()
    logger.info(f"임베딩 생성 완료! 형태: {embeddings.shape}, 소요 시간: {end_time - start_time:.2f} 초")

    try:
        np.save(embedding_filename, embeddings)
        logger.info(f"임베딩을 '{embedding_filename}' 파일로 저장.")
        # ID 저장 시 DataFrame의 인덱스가 아닌 실제 ENTITY_ID 값을 저장해야 함
        ids = df['ENTITY_ID'].values
        np.save(ids_filename, ids)
        logger.info(f"템플릿 ID를 '{ids_filename}' 파일로 저장.")
        return embeddings, ids
    except Exception as e:
        logger.error(f"임베딩/ID 저장 오류: {e}", exc_info=True)
        return None, None

def build_and_save_faiss_index(embeddings, index_filename):
    """Faiss 인덱스 구축 및 저장"""
    if embeddings is None:
        logger.error("임베딩 데이터가 없어 Faiss 인덱스를 구축할 수 없습니다.")
        return None

    logger.info("\nFaiss 인덱스 구축 시작...")
    dimension = embeddings.shape[1]
    logger.info(f"벡터 차원: {dimension}")
    index = faiss.IndexFlatIP(dimension)
    logger.info("IndexFlatIP 인덱스 생성 완료.")

    logger.info(f"{embeddings.shape[0]}개의 벡터를 인덱스에 추가합니다...")
    start_time = time.time()
    index.add(embeddings.astype('float32')) # Faiss는 float32 요구
    end_time = time.time()
    logger.info(f"벡터 추가 완료. 소요 시간: {end_time - start_time:.2f} 초")
    logger.info(f"인덱스 내 총 벡터 수: {index.ntotal}")

    try:
        faiss.write_index(index, index_filename)
        logger.info(f"Faiss 인덱스를 '{index_filename}' 파일로 저장.")
        return index
    except Exception as e:
        logger.error(f"Faiss 인덱스 저장 오류: {e}", exc_info=True)
        return None

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 데이터 로드 및 정제
    template_df = load_and_clean_templates(TEMPLATE_FILEPATH)

    if template_df is not None:
        # --- !!! CSV 파일 저장 (검증용) !!! ---
        try:
            # 저장 전에 DataFrame 정보 로깅
            logger.info(f"저장될 DataFrame 정보:\n{template_df.info()}")
            logger.debug(f"저장될 DataFrame 샘플:\n{template_df.head().to_string()}")
            template_df.to_csv(DEBUG_CSV_FILENAME, index=False, encoding='utf-8-sig')
            logger.info(f"DataFrame 데이터를 '{DEBUG_CSV_FILENAME}'으로 저장했습니다.")
        except Exception as e:
            logger.error(f"CSV 파일 저장 오류: {e}", exc_info=True)
        # --- !!! CSV 저장 끝 !!! ---

        # 2. 임베딩 생성 및 저장
        template_embeddings, template_ids = generate_and_save_embeddings(
            template_df, EMBEDDING_MODEL_NAME, EMBEDDING_FILENAME, IDS_FILENAME
        )

        if template_embeddings is not None:
            # 3. Faiss 인덱스 구축 및 저장
            build_and_save_faiss_index(template_embeddings, INDEX_FILENAME)

    logger.info("="*20 + " 데이터 준비 완료 " + "="*20)
# --- 메인 실행 로직 끝 ---