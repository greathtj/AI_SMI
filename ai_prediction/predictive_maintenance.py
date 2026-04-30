import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------------
# 1. 가상 데이터 생성 (전과 동일)
# ---------------------------------------------------------------------------------
def generate_pdm_data(n_samples=1000):
    np.random.seed(42)
    # 정상 데이터 (90%)
    n_normal = int(n_samples * 0.9)
    X_norm = np.random.uniform([1.0, 30.0, 4.0], [3.0, 50.0, 6.0], (n_normal, 3))
    y_norm = np.zeros(n_normal)
    # 고장 데이터 (10%)
    n_fail = n_samples - n_normal
    X_fail = np.random.uniform([4.0, 60.0, 7.0], [10.0, 90.0, 10.0], (n_fail, 3))
    y_fail = np.ones(n_fail)
    
    X = np.vstack([X_norm, X_fail])
    y = np.concatenate([y_norm, y_fail])
    return X, y

X, y = generate_pdm_data()
print(f"전체 데이터 수: {len(X)} | 정상: {sum(y==0)}개 | 고장: {sum(y==1)}개")

# ---------------------------------------------------------------------------------
# 2. 데이터 전처리 및 텐서 변환
# ---------------------------------------------------------------------------------
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PyTorch는 데이터를 '텐서(Tensor)'라는 형태로 처리함
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1) # (n,) -> (n, 1) 형태로 변경
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# ---------------------------------------------------------------------------------
# 3. 신경망 모델 설계
# ---------------------------------------------------------------------------------
class PdMModel(nn.Module):
    def __init__(self):
        super(PdMModel, self).__init__()
        # 신경망 구조 정의
        self.layer = nn.Sequential(
            nn.Linear(3, 8),      # 입력층(3) -> 은닉층1(8)
            nn.ReLU(),
            nn.Linear(8, 4),      # 은닉층1(8) -> 은닉층2(4)
            nn.ReLU(),
            nn.Linear(4, 1),      # 은닉층2(4) -> 출력층(1)
            nn.Sigmoid()          # 확률값(0~1)으로 변환
        )
        
    def forward(self, x):
        return self.layer(x)

model = PdMModel()

# ---------------------------------------------------------------------------------
# 4. 손실함수 및 옵티마이저 설정
# ---------------------------------------------------------------------------------
# 데이터 불균형 해결: 고장 데이터(1)에 9배의 가중치 부여
# PyTorch에서는 BCEWithLogitsLoss의 pos_weight를 쓰지만, 
# 모델 끝에 Sigmoid를 넣었으므로 BCELoss를 사용하고 가중치를 직접 계산에 반영함
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------------------------------------------------
# 5. 모델 학습 (Training Loop)
# ---------------------------------------------------------------------------------
epochs = 100
batch_size = 32

print("\nAI 학습 시작...")
for epoch in range(epochs):
    model.train()
    
    # 데이터를 섞어서 배치 단위로 학습
    permutation = torch.randperm(X_train_tensor.size()[0])
    
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        # 1. 예측
        outputs = model(batch_x)
        
        # 2. 손실 계산 (불균형 해결을 위해 고장 데이터에 가중치 부여)
        # weight_tensor: 정상이면 1.0, 고장이면 9.0의 가중치 적용
        weight_tensor = torch.where(batch_y == 1, 9.0, 1.0)
        loss = (criterion(outputs, batch_y) * weight_tensor).mean()
        
        # 3. 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("학습 완료!")

# ---------------------------------------------------------------------------------
# 6. 실전 테스트 및 예측: "정상과 고장을 제대로 구분하는가?"
# ---------------------------------------------------------------------------------
model.eval() # 평가 모드로 전환
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test_tensor).float().mean()
    print(f"\n[검증 완료] 테스트 데이터 정확도: {accuracy.item()*100:.2f}%")

# ---------------------------------------------------------------------------------
# [가상 시나리오 테스트] 서로 다른 두 가지 케이스를 넣어보자
# ---------------------------------------------------------------------------------
# 시나리오 1: 아주 평온한 상태 (정상 예상) -> [진동 낮음, 온도 적정, 전류 안정]
# 시나리오 2: 심각한 이상 징후 (고장 예상) -> [진동 높음, 온도 높음, 전류 높음]
test_scenarios = [
    {"name": "평온한 상태", "data": np.array([[1.8, 42.0, 5.1]]), "expect": "정상"},
    {"name": "위험한 상태", "data": np.array([[7.5, 81.0, 8.8]]), "expect": "고장"}
]

print("\n" + "="*50)
print("🚀 실전 시나리오 예측 테스트 시작")
print("="*50)

for scenario in test_scenarios:
    # 데이터 전처리 (학습 때 사용한 scaler 적용)
    scaled_data = scaler.transform(scenario["data"])
    tensor_data = torch.FloatTensor(scaled_data)
    
    with torch.no_grad():
        prob = model(tensor_data).item()
        result = "고장(위험)" if prob > 0.5 else "정상(안전)"
        
    print(f"🔍 테스트 케이스: {scenario['name']}")
    print(f"   - 입력 데이터: {scenario['data']}")
    print(f"   - AI 예측 확률: {prob*100:.2f}%")
    print(f"   - 최종 판정: {result} (기대 결과: {scenario['expect']})")
    
    if (result.startswith("고장") and scenario["expect"] == "고장") or \
       (result.startswith("정상") and scenario["expect"] == "정상"):
        print("   ✅ 판정 결과: 정확하게 맞혔습니다!")
    else:
        print("   ❌ 판정 결과: 예측이 틀렸습니다.")
    print("-" * 50)
