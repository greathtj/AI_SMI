import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 가상 데이터 생성 (단일 변수: 온도 데이터 500개)
np.random.seed(42)
torch.manual_seed(42)

# 정상 데이터: 평균 205도, 표준편차 2도의 온도 데이터
normal_temp = np.random.normal(loc=205, scale=2, size=(500, 1))

# 이상치 데이터 생성 (5개)
outlier_temp = np.array([
    [225], # 너무 높음
    [185], # 너무 낮음
    [230], # 아주 높음
    [190], # 낮음
    [218]  # 경계선상의 높은 온도
])

# 데이터 합치기
all_data = np.vstack([normal_temp, outlier_temp])

# 데이터 무작위 섞기
indices = np.arange(len(all_data))
np.random.shuffle(indices)
all_data = all_data[indices]

# 데이터 스케일링 (0~1 사이 값으로 변환)
scaler = MinMaxScaler()
all_data_scaled = scaler.fit_transform(all_data).astype(np.float32)

# 정상 데이터만 골라내어 학습용 데이터로 사용
is_normal = (all_data >= 195) & (all_data <= 215)
train_data_scaled = all_data_scaled[is_normal.flatten()]

train_data_tensor = torch.tensor(train_data_scaled)
train_loader = DataLoader(TensorDataset(train_data_tensor), batch_size=16, shuffle=True)

# 2. PyTorch 오토인코더 모델 설계 (1차원 입력 -> 1차원 압축 -> 1차원 복원)
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # 인코더: 데이터를 더 작은 특징으로 압축
        self.encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        # 디코더: 데이터를 다시 복원
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = SimpleAutoencoder()

# 손실 함수와 최적화 도구
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 모델 학습
epochs = 150
model.train()
print("단일 변수(온도) 모델 학습 중...")

for epoch in range(epochs):
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 4. 복원 및 오차 계산
model.eval()
with torch.no_grad():
    all_inputs = torch.tensor(all_data_scaled)
    reconstructed_scaled = model(all_inputs)
    mse = torch.mean((all_inputs - reconstructed_scaled)**2, dim=1).numpy()
    reconstructed_data = scaler.inverse_transform(reconstructed_scaled.numpy())

# 5. 결과 시각화
plt.figure(figsize=(14, 15))

# [그래프 1] 순수 원본 데이터 분포 (추가됨)
plt.subplot(3, 1, 1)
sample_range = range(len(all_data))
plt.scatter(sample_range, all_data, color='gray', alpha=0.5, s=15, label='Collected Raw Data')
actual_outlier_indices = np.where(~is_normal.flatten())[0]
plt.scatter(actual_outlier_indices, all_data[actual_outlier_indices], color='red', s=60, label='Hidden Outliers', zorder=5)
plt.title('Step 1: Raw Temperature Data (Before Analysis)')
plt.ylabel('Temperature (°C)')
plt.legend()

# [그래프 2] 원본 온도 vs 복원된 온도 비교
plt.subplot(3, 1, 2)
plt.plot(sample_range, all_data, color='gray', alpha=0.4, label='Original Input', linewidth=1)
plt.plot(sample_range, reconstructed_data, color='blue', alpha=0.7, label='Reconstructed Output', linewidth=1)
plt.scatter(actual_outlier_indices, all_data[actual_outlier_indices], color='red', s=60, label='Actual Outliers', zorder=5)
plt.title('Step 2: Comparison - Original vs Reconstructed')
plt.ylabel('Temperature (°C)')
plt.legend()

# [그래프 3] 복원 오차(MSE)와 임계값
plt.subplot(3, 1, 3)
plt.bar(sample_range, mse, color='skyblue', alpha=0.8, label='Reconstruction Error')
threshold = np.sort(mse)[-6] # 상위 5개를 잡기 위한 임계값
plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
plt.title('Step 3: Quantifying Anomaly via Reconstruction Error')
plt.xlabel('Data Index')
plt.ylabel('MSE (Error)')
plt.legend()

plt.tight_layout()
plt.show()

# 상세 비교 출력
print("\n--- 이상치 데이터 상세 비교 (단일 변수) ---")
for idx in actual_outlier_indices:
    print(f"인덱스 {idx}: 원본 {all_data[idx][0]:.2f}°C -> 복원 {reconstructed_data[idx][0]:.2f}°C (오차: {mse[idx]:.4f})")
