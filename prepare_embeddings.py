import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

def prepare_embeddings():
    # FAQ 데이터 로드
    df = pd.read_excel('data/faq.xlsx')
    
    # 임베딩 모델 로드
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    
    # FAQ 임베딩 생성
    faq_embeddings = model.encode(df['question'].tolist())
    
    # FAISS 인덱스 생성
    dimension = faq_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(faq_embeddings.astype('float32'))
    
    # 결과 저장
    faiss.write_index(index, 'data/faq_index.faiss')
    df.to_json('data/faq_data.json', orient='records', force_ascii=False)
    
    print("임베딩 및 인덱스 생성 완료")

if __name__ == "__main__":
    prepare_embeddings() 