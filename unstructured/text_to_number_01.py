import pandas as pd
from collections import Counter
import re

# 1. 비정형 텍스트 데이터 (예: 제조 현장 작업 일지 또는 고객 피드백)
raw_text_data = [
    "Machine overheating issues in section A",
    "Request for maintenance on sensor B",
    "Section A machine is running normally now",
    "Sensor B failure detected again",
    "Emergency shutdown in section C due to overheating",
    "Routine check completed for all machines"
]

def process_unstructured_text(text_list):
    # 모든 문장을 하나로 합치고 소문자로 변환 (전처리)
    all_text = " ".join(text_list).lower()
    
    # 특수문자 제거 및 단어 단위 분리 (비정형 -> 정형화 단계)
    words = re.findall(r'\w+', all_text)
    
    # 단어 빈도수 계산
    word_counts = Counter(words)
    
    # 결과를 판다스 데이터프레임으로 변환 (정형 데이터화)
    df = pd.DataFrame(word_counts.items(), columns=['Keyword', 'Frequency'])
    
    # 빈도수 순으로 정렬
    return df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

# 2. 변환 실행
structured_df = process_unstructured_text(raw_text_data)

# 3. 결과 출력
print("--- Text Data to Structured Data Transformation ---")
print(structured_df.head(10))

# 특정 핵심 키워드(오버히팅 등)가 얼마나 발생했는지 즉각적인 분석이 가능해집니다.
overheating_count = structured_df[structured_df['Keyword'] == 'overheating']['Frequency'].values[0]
print(f"\n[Insight] 'Overheating' mentioned {overheating_count} times in unstructured logs.")
