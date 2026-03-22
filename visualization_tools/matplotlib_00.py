import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 1. 시각화 데이터 생성 (압력 센서 데이터 시나리오)
np.random.seed(10)
time = np.linspace(0, 10, 100)
# 정상 흐름에 미세한 노이즈 추가
pressure = 30 + np.sin(time) * 2 + np.random.normal(0, 0.5, 100)

# 이상 상황 발생 구간 강제 설정 (6~8시간 사이 압력 급증)
pressure[60:80] = pressure[60:80] + 8 + np.random.normal(0, 2, 20)

# 2. 도화지(Figure)와 축(Axes) 설정
# 인쇄용 보고서에 적합한 고해상도 설정을 가정합니다.
fig, ax = plt.subplots(figsize=(12, 7), facecolor='#ffffff')
plt.subplots_adjust(top=0.85, bottom=0.15)

# 3. 메인 그래프 그리기
line, = ax.plot(time, pressure, color='#1e293b', linewidth=2, label='Real-time Pressure')

# 4. Matplotlib의 '자유도'를 활용한 인포그래픽 요소 추가
# (1) 관리 임계치 영역 강조 (박스 형태)
rect = patches.Rectangle((0, 25), 10, 10, linewidth=0, facecolor='#3b82f6', alpha=0.1, zorder=0)
ax.add_patch(rect)
ax.text(0.2, 33, 'Normal Operating Range (25-35 Bar)', color='#3b82f6', fontweight='bold', fontsize=10)

# (2) 이상 발생 구간 화살표 및 텍스트 상자 (Annotation)
# Matplotlib은 화살표 하나도 정교하게 커스터마이징 가능합니다.
ax.annotate('Critical Pressure Spike!', 
            xy=(7, 40), xytext=(8, 45),
            arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=2, headwidth=10),
            bbox=dict(boxstyle='round,pad=0.5', fc='#fee2e2', ec='#ef4444', lw=1),
            fontsize=11, color='#b91c1c', fontweight='bold')

# (3) 하단 상태 표시 바 (Infographic Style)
ax.fill_between(time[60:82], 20, 55, color='#ef4444', alpha=0.15, zorder=1)
ax.text(7, 22, '▼ Emergency Check Required', color='#ef4444', ha='center', fontweight='bold')

# 5. 축 및 디자인 다듬기 (설계 도면 스타일)
ax.set_title('LINE-04 Hydraulic Pressure Infographic Report', fontsize=20, fontweight='black', pad=30, loc='left')
ax.set_xlabel('Operation Time (Hours)', fontsize=12, labelpad=10)
ax.set_ylabel('Pressure (Bar)', fontsize=12, labelpad=10)

# 불필요한 테두리 제거 및 그리드 설정
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Y축 범위 조정 및 틱 설정
ax.set_ylim(20, 55)
ax.set_xlim(0, 10)

# 6. 상단 요약 정보 (Text-based Infographic)
plt.figtext(0.125, 0.85, "Status: WARNING", fontsize=14, color='white', 
            bbox=dict(facecolor='#ef4444', edgecolor='none', boxstyle='round,pad=0.3'))
plt.figtext(0.28, 0.85, "Avg: 32.4 Bar | Max: 43.1 Bar | Sensor: P-901", fontsize=12, color='#64748b')

# 7. 참고: 센서 제작에 프로그래밍 기술은 필요 없다는 메시지 (User Request 반영)
plt.figtext(0.5, 0.02, "* Note: Building these monitoring sensors requires NO programming skills.", 
            ha='center', fontsize=10, color='#94a3b8', fontstyle='italic')

plt.show()
