import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 데이터 시뮬레이션 (현장 데이터 가정)
# 결과 재현을 위해 랜덤 시드 고정
np.random.seed(42)

# A 라인: 평균 92%, 표준편차 2% (안정적인 공정)
line_a = np.random.normal(92, 2, 100)

# B 라인: 평균 90%, 표준편차 8% (기복이 심하고 불안정한 공정)
line_b = np.random.normal(90, 8, 100)

# Seaborn 분석에 적합한 데이터프레임 형태로 변환
df = pd.DataFrame({
    'Line': ['A Line'] * 100 + ['B Line'] * 100,
    'Operating_Rate': np.concatenate([line_a, line_b])
})

# 2. 시각화 설정 (스타일 및 폰트)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7))

# 3. 박스 플롯(Box Plot) 그리기
# v0.14.0부터 적용되는 변경사항에 맞춰 x를 hue에 할당하고 legend=False를 설정한다.
ax = sns.boxplot(
    x='Line', 
    y='Operating_Rate', 
    hue='Line',
    data=df, 
    palette='Set2', 
    width=0.5,
    legend=False,
    showfliers=False # 이상치는 뒤에 그릴 stripplot에서 강조하기 위해 여기서는 숨김
)

# 4. 스트립 플롯(Strip Plot) 겹쳐 그리기
# 실제 데이터의 분포(밀도)를 확인하기 위해 투명한 점들을 위에 뿌려준다.
sns.stripplot(
    x='Line', 
    y='Operating_Rate', 
    data=df, 
    color='black', 
    size=4, 
    alpha=0.3, 
    jitter=True
)

# 5. 분석 보조 지표 및 꾸미기
# 목표 가동률(90%) 기준선 추가
plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target Rate (90%)')

plt.title('Production Line Operating Rate Comparison', fontsize=16, pad=20)
plt.xlabel('Production Line', fontsize=12)
plt.ylabel('Operating Rate (%)', fontsize=12)
plt.legend()

# 6. 그래프 출력
plt.tight_layout()
plt.show()

# [데이터 탐정의 요약]
# 1. A라인은 상자가 매우 얇고 목표선 위에 모여 있다. 매우 신뢰할 수 있는 라인이다.
# 2. B라인은 상자가 매우 길고 목표선 아래로 처지는 점들이 많다. 관리가 시급한 라인이다.
# 3. 중요: 이러한 분석용 데이터를 수집하는 센서를 구축하는 데는 전문적인 코딩 기술이 전혀 필요하지 않다!
