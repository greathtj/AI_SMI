import pandas as pd
import numpy as np

# 1. 가상의 주조 공정 데이터 생성
# 설비명, 측정시간, 온도로 구성된 데이터셋 (일부러 결측치 NaN을 삽입)
data = {
    '설비ID': ['Furnace_A', 'Furnace_A', 'Furnace_A', 'Furnace_A', 'Furnace_B', 'Furnace_B', 'Furnace_B', 'Furnace_B'],
    '측정시간': ['10:00', '10:05', '10:10', '10:15', '10:00', '10:05', '10:10', '10:15'],
    '온도': [1100, 1105, np.nan, 1110, 1500, np.nan, 1510, 1520] # NaN이 결측치
}

df = pd.DataFrame(data)
print("--- [결측치가 포함된 원본 데이터] ---")
print(df)
print("\n")

# 2. [위험한 방법] 전체 평균으로 채우기
# 모든 설비의 온도를 다 합쳐서 평균을 낸 뒤 채우는 방식 (추천하지 않음)
overall_mean = df['온도'].mean()
df_wrong = df.copy()
df_wrong['온도'] = df_wrong['온도'].fillna(overall_mean)

print(f"전체 평균값: {overall_mean:.2f}")
print("--- [전체 평균으로 채운 결과 - 위험!] ---")
print(df_wrong) 
# 결과: Furnace_A(1100대)와 B(1500대)의 중간값인 약 1300대가 들어가게 됨 -> 데이터 왜곡 발생
print("\n")

# 3. [실무적 방법] 설비별(그룹별) 평균으로 채우기
# 설비ID별로 그룹을 나누어, 각 그룹의 평균값으로 해당 그룹의 결측치를 채움
df_correct = df.copy()
df_correct['온도'] = df_correct.groupby('설비ID')['온도'].transform(lambda x: x.fillna(x.mean()))

print("--- [설비별 평균으로 채운 결과 - 권장] ---")
print(df_correct)
# 결과: Furnace_A의 결측치는 A의 평균으로, B의 결측치는 B의 평균으로 정교하게 채워짐
