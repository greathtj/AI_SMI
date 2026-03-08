import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 시뮬레이션 데이터 생성 (실제 현장에서는 엑셀/CSV 파일을 불러오게 됩니다)
# 가동 시간(Runtime)은 수만 단위, 불량률(Defect Rate)은 0.001 단위의 미세 수치입니다.
data = {
    'Day': [f'Day {i+1}' for i in range(15)],
    'Runtime_Min': [10000, 15000, 22000, 31000, 40000, 48000, 55000, 62000, 71000, 78000, 85000, 92000, 95000, 98000, 100000],
    'Defect_Rate_Pct': [0.002, 0.005, 0.003, 0.008, 0.012, 0.025, 0.045, 0.088, 0.150, 0.320, 0.750, 1.250, 2.500, 5.800, 12.500]
}

df = pd.DataFrame(data)

# 2. 시각화 설정 (한 화면에 두 가지 스케일 비교)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.3)

# --- 첫 번째 그래프: 일반 스케일 (Linear Scale) ---
# 불량률이 너무 작아서 바닥에 붙어 보이며, 추이를 확인하기 어렵습니다.
ax1.plot(df['Day'], df['Runtime_Min'], color='blue', marker='o', label='Runtime (min)')
ax1.plot(df['Day'], df['Defect_Rate_Pct'], color='red', marker='s', label='Defect Rate (%)')
ax1.set_title('Linear Scale: Large vs Small Data (Hard to compare)')
ax1.set_ylabel('Absolute Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 두 번째 그래프: 로그 스케일 (Logarithmic Scale) ---
# 로그 스케일을 적용하면 큰 수치와 미세 수치를 동일 평면에서 명확히 비교할 수 있습니다.
ax2.plot(df['Day'], df['Runtime_Min'], color='blue', marker='o', label='Runtime (min)')
ax2.plot(df['Day'], df['Defect_Rate_Pct'], color='red', marker='s', label='Defect Rate (%)')
ax2.set_yscale('log')  # 핵심 코드: Y축을 로그 스케일로 변환
ax2.set_title('Logarithmic Scale: Clear Correlation Visualization')
ax2.set_ylabel('Value (Log Scale)')
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)

# 3. 분석 결과 출력
print("--- Manufacturing Data Analysis ---")
print(df.describe())
print("\n[Insight] As runtime approaches 100,000 min, defect rates show exponential growth.")

plt.show()
