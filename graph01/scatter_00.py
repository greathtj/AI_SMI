import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 실습용 가상 데이터 생성
# 현장 상황 가정: 가공 온도(X)가 올라가면 제품의 치수(Y)가 커지는 양의 상관관계
np.random.seed(42)
temp = np.random.uniform(150, 250, 50)  # 150도~250도 사이의 온도 데이터 50개
size = 0.5 * temp + 10 + np.random.normal(0, 5, 50)  # 온도에 따른 치수 변화 + 약간의 노이즈

# 2. 그래프 기본 설정 (한글 깨짐 방지 설정이 필요할 수 있음)
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid') # 깔끔한 화이트 그리드 스타일 적용

# 3. 산점도 그리기 (Scatter Plot)
# s는 점의 크기, alpha는 투명도, edgecolors는 점의 테두리 설정
plt.scatter(temp, size, color='dodgerblue', s=80, alpha=0.7, edgecolors='w', label='Measured Data')

# 4. 추세선(Regression Line) 추가하기
# 1차 방정식(y = ax + b) 형태로 데이터를 가장 잘 설명하는 선을 계산
z = np.polyfit(temp, size, 1) # 계수 계산
p = np.poly1d(z) # 방정식 생성
plt.plot(temp, p(temp), "r--", linewidth=2, label='Trend Line (Correlation)')

# 5. 그래프 정보 추가 (축 이름, 제목)
plt.title('Process Analysis: Temperature vs. Product Size', fontsize=15, pad=15)
plt.xlabel('Processing Temperature (Celsius)', fontsize=12)
plt.ylabel('Product Dimension (mm)', fontsize=12)

# 6. 범례 및 레이아웃 조정
plt.legend()
plt.tight_layout()

# 그래프 출력
plt.show()

# ---------------------------------------------------------
# [추가 팁] Seaborn 라이브러리를 사용하면 단 한 줄로도 가능하다!
# ---------------------------------------------------------
# plt.figure(figsize=(10, 6))
# sns.regplot(x=temp, y=size, scatter_kws={'s':80, 'alpha':0.6}, line_kws={'color':'red'})
# plt.show()
