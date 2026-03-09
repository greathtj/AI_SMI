import pandas as pd
import numpy as np
from datetime import datetime

def generate_sample_data():
    """
    분석을 위한 가상의 시뮬레이션 데이터 생성
    """
    # 1. 고빈도 센서 데이터 (1초 단위 측정)
    timestamps = pd.date_range("2023-10-01 10:00:00", periods=300, freq='s')
    sensor_df = pd.DataFrame({
        'timestamp': timestamps,
        'injection_pressure': np.random.uniform(140, 160, 300), # 사출 압력 (bar)
        'mold_temperature': np.random.uniform(40, 50, 300),     # 금형 온도 (℃)
        'cycle_time': np.random.uniform(15, 20, 300)            # 사이클 타임 (sec)
    })

    # 2. 품질 검사 데이터 (비정기적 발생)
    quality_df = pd.DataFrame({
        'inspection_time': [
            pd.Timestamp("2023-10-01 10:01:00"),
            pd.Timestamp("2023-10-01 10:02:00"),
            pd.Timestamp("2023-10-01 10:03:00"),
            pd.Timestamp("2023-10-01 10:04:00")
        ],
        'is_defective': [0, 1, 0, 0] # 0: 정상, 1: 불량
    })
    
    return sensor_df, quality_df

def create_analysis_dataset(sensor_data, quality_data, window_seconds=30):
    """
    품질 검사 시점을 기준으로 과거 N초간의 센서 데이터를 집계하여 통합
    """
    combined_results = []

    for _, q_row in quality_data.iterrows():
        end_time = q_row['inspection_time']
        start_time = end_time - pd.Timedelta(seconds=window_seconds)

        # 시간 윈도우 필터링
        mask = (sensor_data['timestamp'] >= start_time) & (sensor_data['timestamp'] <= end_time)
        relevant_sensors = sensor_data.loc[mask]

        if not relevant_sensors.empty:
            features = {
                'inspection_time': end_time,
                'avg_pressure': relevant_sensors['injection_pressure'].mean(),
                'max_temp': relevant_sensors['mold_temperature'].max(),
                'avg_cycle': relevant_sensors['cycle_time'].mean(),
                'label_defective': q_row['is_defective']
            }
            combined_results.append(features)

    return pd.DataFrame(combined_results)

# --- 메인 실행부 ---
if __name__ == "__main__":
    # 1. 원본 데이터 생성
    raw_sensor, raw_quality = generate_sample_data()

    # 2. 원본 데이터 출력 (상위 5개씩)
    print("-" * 50)
    print("[원본 데이터 1] 센서 데이터 (실시간 수집)")
    print(raw_sensor.head())
    print(f"\n총 데이터 개수: {len(raw_sensor)}건")
    
    print("-" * 50)
    print("[원본 데이터 2] 품질 검사 데이터 (현장 기록)")
    print(raw_quality)
    print("-" * 50)

    # 3. 데이터 통합 수행
    analysis_df = create_analysis_dataset(raw_sensor, raw_quality, window_seconds=30)

    # 4. 최종 통합 데이터 출력
    print("\n[최종 결과] 불량 원인 추적을 위한 분석용 데이터셋")
    print("(검사 시점 직전 30초간의 센서 데이터 통계량 결합)")
    print(analysis_df)
    print("-" * 50)
