import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 가상의 데이터 생성 (작업 시간 대비 생산량)
# 생산량은 로그 함수를 사용하여 일정 시간이 지나면 포화(Saturation)되는 형태를 모사한다.
np.random.seed(42)
n_samples = 150
work_hours = np.random.uniform(1, 15, n_samples)
# 생산량 모델: 20 * log(work_hours + 1) + 노이즈
production = 20 * np.log(work_hours + 1) + np.random.normal(0, 1.2, n_samples)

df = pd.DataFrame({'Work_Hours': work_hours, 'Production': production})

# 결측치 생성: 고부하 작업 시간대(10시간 이상) 데이터 누락 가정
missing_idx = df[df['Work_Hours'] > 10].sample(frac=0.7).index
df_missing = df.copy()
df_missing.loc[missing_idx, 'Work_Hours'] = np.nan

# 결측치를 평균값으로 채운 데이터셋 생성
df_imputed = df_missing.copy()
df_imputed['Work_Hours'] = df_imputed['Work_Hours'].fillna(df_imputed['Work_Hours'].mean())

# 시각화를 위한 설정 (한글 깨짐 방지를 위해 영문 레이블 사용)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- [사례 1: 모델 학습 실패 (Error Propagation)] ---
print("--- Case 1: Model Training Attempt ---")
try:
    model_fail = LinearRegression()
    # 결측치가 있는 데이터로 학습 시도
    model_fail.fit(df_missing[['Work_Hours']], df_missing['Production'])
except ValueError as e:
    error_msg = str(e)
    print(f"Error Occurred: {error_msg}")
    
    # 텍스트로 시각화 영역 표시
    axes[0].text(0.5, 0.5, f"ValueError:\nInput contains NaN\n\nML Model Stopped", 
                ha='center', va='center', color='red', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=1'))
    axes[0].set_title("1. Model Training Failure", fontsize=14, fontweight='bold')
    axes[0].axis('off')

# --- [사례 2: 작업 시간 분포 왜곡 (Statistical Bias)] ---
sns.kdeplot(df['Work_Hours'], label='Original (Actual)', fill=True, color='blue', ax=axes[1])
sns.kdeplot(df_imputed['Work_Hours'], label='Mean Imputed', fill=True, color='red', ax=axes[1])
axes[1].axvline(df_imputed['Work_Hours'].mean(), color='red', linestyle='--', alpha=0.6)
axes[1].set_title("2. Work Hours Distribution Bias", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Work Hours")
axes[1].set_ylabel("Density")
axes[1].legend()

# --- [사례 3: Saturation 도달 학습 왜곡 (Relationship Distortion)] ---
# 원본 데이터 흐름 (비선형 관계)
axes[2].scatter(df['Work_Hours'], df['Production'], alpha=0.3, color='blue', label='Actual Relationship')

# 결측치였다가 평균으로 채워진 데이터 (왜곡된 포인트)
imputed_points = df_imputed.loc[missing_idx]
axes[2].scatter(imputed_points['Work_Hours'], imputed_points['Production'], 
                color='orange', marker='x', s=100, label='Distorted Data (Mean Imputed)')

# 선형 회귀 비교 (원본 vs 왜곡)
# 원본 학습 (NaN 제외)
model_orig = LinearRegression().fit(df[['Work_Hours']], df['Production'])
# 왜곡 학습 (평균 보간)
model_distort = LinearRegression().fit(df_imputed[['Work_Hours']], df_imputed['Production'])

# UserWarning 방지를 위해 DataFrame 형태로 예측 데이터 생성
line_x = pd.DataFrame({'Work_Hours': np.linspace(1, 15, 100)})
axes[2].plot(line_x, model_orig.predict(line_x), color='blue', linestyle='--', label='Actual Trend')
axes[2].plot(line_x, model_distort.predict(line_x), color='red', linewidth=2, label='Distorted Trend')

axes[2].set_title("3. Saturation Learning Distortion", fontsize=14, fontweight='bold')
axes[2].set_xlabel("Work Hours")
axes[2].set_ylabel("Production")
axes[2].legend()

plt.tight_layout()
plt.show()

print("\n[분석 요약]")
print("1. 학습 실패: NaN(결측치) 값이 포함된 데이터는 연산 엔진을 중단시킨다다.")
print("2. 분포 왜곡: 단순 평균값 보간은 데이터의 변동성을 인위적으로 줄여 통계적 편향을 초래한다.")
print("3. 학습 왜곡: 고부하(포화) 영역의 데이터를 평균값으로 대체하면 모델이 공정의 한계를 예측하는 능력을 상실한다.")
