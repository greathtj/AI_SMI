import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 만들기 (개미 99명과 거인 1명)
ant_money = np.random.uniform(1000, 50000, 99) # 1,000원 ~ 50,000원 사이
giant_money = [1000000000] # 10억 원 (거인)
all_money = np.concatenate([ant_money, giant_money])

# 2. 로그 변환하기
log_money = np.log10(all_money)

# 3. 그림으로 비교하기
plt.figure(figsize=(12, 5))

# [왼쪽] 원본 그래프: 거인 때문에 개미들이 바닥에 붙어있음
plt.subplot(1, 2, 1)
plt.hist(all_money, bins=50, color='gray')
plt.title("Original: Ants are invisible!")
plt.xlabel("Money")
plt.ylabel("Count")

# [오른쪽] 로그 그래프: 개미들의 분포가 선명하게 보임!
plt.subplot(1, 2, 2)
plt.hist(log_money, bins=20, color='orange', edgecolor='black')
plt.title("Log Transformed: Now we see everyone!")
plt.xlabel("Log10(Money)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
