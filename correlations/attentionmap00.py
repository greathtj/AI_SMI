import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 가상 어텐션 데이터 생성 (AI의 시선 추적)
# 특정 불량 판정 시점에 AI가 각 센서를 얼마나 중요하게 봤는지(Attention Weight) 설정합니다.
# 0에서 1 사이의 값을 가지며, 높을수록 AI가 그 부분을 집중해서 본 것입니다.

time_steps = ['T-4 (5분 전)', 'T-3 (4분 전)', 'T-2 (3분 전)', 'T-1 (2분 전)', 'Now (현재)']
sensors = ['Temperature', 'Pressure', 'Vibration', 'Humidity']

# AI가 현재(Now) 시점의 'Vibration(진동)' 센서에 시선이 꽂혔다고 가정하는 데이터
# 마지막 행(Now)의 Vibration 항목에 높은 점수(0.92)를 부여합니다.
attention_weights = np.array([
    [0.10, 0.05, 0.15, 0.10],
    [0.12, 0.08, 0.20, 0.10],
    [0.15, 0.10, 0.35, 0.12],
    [0.08, 0.12, 0.65, 0.15],
    [0.02, 0.03, 0.92, 0.03]  # AI의 눈동자가 진동(Vibration)에 고정됨!
])

# 데이터프레임으로 변환
df_attention = pd.DataFrame(attention_weights, columns=sensors, index=time_steps)

# 2. 시각화: AI의 눈동자 추적 지도 (어텐션 맵)
plt.figure(figsize=(12, 7))

# sns.heatmap 설정 설명:
# annot=True: 칸 안에 숫자를 표시
# cmap='YlOrRd': 노란색에서 빨간색으로 (빨간색일수록 집중도가 높음)
sns.heatmap(df_attention, annot=True, fmt=".2f", cmap='YlOrRd', linewidths=1, linecolor='white')

plt.title("Attention Map: AI's Focus Investigation\n(Where is AI looking right now?)", fontsize=16)
plt.ylabel("Time Flow (Back to Present)")
plt.xlabel("Sensor Type")
plt.show()

# 3. 분석 결과 해석 (평서체)
print("--- [데이터 수사 보고서: AI의 속마음 읽기] ---")
print("1. 어텐션 맵의 가장 진한 빨간색(0.92)을 확인하라.")
print("   - 현재(Now) 시점에서 'Vibration(진동)' 센서의 색깔이 가장 뜨겁다.")
print("   - 인공지능은 지금 불량을 판정하기 위해 진동 데이터를 가장 유심히 째려보고 있다.")

print("\n2. 시간의 흐름을 분석하라.")
print("   - 3분 전(T-2)부터 진동에 대한 AI의 관심이 0.35 -> 0.65 -> 0.92로 점점 높아졌다.")
print("   - 즉, 갑자기 터진 불량이 아니라 몇 분 전부터 조짐이 있었다는 것을 AI는 알고 있었다.")

print("\n3. 다른 센서들은 상대적으로 연한 색이다.")
print("   - 온도나 습도는 현재 상황에서 AI가 보기에 '범인이 아님'을 뜻한다.")

print("\n[결론] 어텐션 맵은 AI가 내린 결론의 '증거'를 시각적으로 보여주는 강력한 도구다.")

print("\n[팁] 우리가 직접 만든 센서 데이터를 입력하면, 현대적인 AI 분석 도구들이")
print("이런 '수사관의 시선'을 자동으로 그려주어 여러분을 공장의 도사로 만들어 줄 것이다!")
