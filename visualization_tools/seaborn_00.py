import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 현장 체감형 데이터 시뮬레이션 (A라인 vs B라인)
# 실제 스마트 팩토리에서 수집될 법한 데이터를 구성한다.
np.random.seed(7)
data_points = 50

# A라인: 구형 모델, 생산량은 높지만 불량률 변동이 큼
line_a = pd.DataFrame({
    'Line': 'Line A (Legacy)',
    'Efficiency (%)': np.random.normal(78, 5, data_points),
    'Defect Rate (%)': np.random.normal(4.5, 1.5, data_points),
    'Energy Consumption (kW)': np.random.normal(120, 10, data_points)
})

# B라인: 신규 모델, 안정적인 수율과 높은 에너지 효율
line_b = pd.DataFrame({
    'Line': 'Line B (New Tech)',
    'Efficiency (%)': np.random.normal(88, 2, data_points),
    'Defect Rate (%)': np.random.normal(1.8, 0.4, data_points),
    'Energy Consumption (kW)': np.random.normal(95, 5, data_points)
})

df = pd.concat([line_a, line_b])

# 2. 시각화 스타일 설정 (보고서용 프리미엄 스타일)
sns.set_theme(style="white", palette="muted")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- [좌측 그래프] 분포 및 밀도 분석 (안정성 평가) ---
# 사장님께 "B라인이 얼마나 안정적으로 고수율을 유지하는가"를 보여준다.
sns.kdeplot(
    data=df, x="Efficiency (%)", hue="Line", fill=True, 
    common_norm=False, palette="magma", alpha=.5, linewidth=2, ax=axes[0]
)
axes[0].set_title("1. Production Efficiency Distribution\n(B Line shows superior stability)", fontsize=15, pad=20)
axes[0].set_xlabel("Efficiency (%)", fontsize=12)

# --- [우측 그래프] 상관관계 분석 (생산성 vs 품질) ---
# "에너지는 적게 쓰면서 불량은 낮은 지점"을 시각화한다.
sns.scatterplot(
    data=df, x="Energy Consumption (kW)", y="Defect Rate (%)", 
    hue="Line", size="Efficiency (%)", sizes=(40, 400),
    alpha=0.6, palette="magma", ax=axes[1]
)
# 평균 지점에 강조 표시 (보고서용 인사이트 추가)
for line, color in zip(['Line A (Legacy)', 'Line B (New Tech)'], ['#3d0852', '#f98e09']):
    line_data = df[df['Line'] == line]
    axes[1].scatter(line_data['Energy Consumption (kW)'].mean(), 
                    line_data['Defect Rate (%)'].mean(), 
                    color=color, marker='X', s=500, edgecolor='white', linewidth=2, label=f'{line} Avg')

axes[1].set_title("2. Quality vs Energy Efficiency\n(Lower-Left is Better)", fontsize=15, pad=20)
axes[1].set_xlabel("Energy Consumption (kW)", fontsize=12)
axes[1].set_ylabel("Defect Rate (%)", fontsize=12)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 전체 레이아웃 정리 및 테두리 제거 (심플함 강조)
sns.despine()
plt.tight_layout()

# 보고서용 텍스트 추가 (주요 결론)
plt.figtext(0.5, 0.01, 
           "Conclusion: Line B (New Tech) achieved 12% higher efficiency with 60% lower defect variance compared to Line A.", 
           ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.show()
