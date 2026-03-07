import numpy as np
import matplotlib.pyplot as plt

def extract_fft_features():
    # 1. 데이터 생성 (시간 영역)
    sampling_rate = 1000
    t = np.linspace(0, 1.0, sampling_rate)

    # 두 개의 신호(특징)가 섞인 Raw Data
    # 특징 A: 60Hz 진동 (정상 가동)
    # 특징 B: 120Hz 미세 진동 (초기 결함 징후)
    raw_signal = np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    noise = np.random.normal(0, 0.5, len(t))
    data = raw_signal + noise

    # 2. FFT를 통한 특성 공학 (Feature Engineering)
    n = len(data)
    fft_val = np.fft.fft(data)
    freqs = np.fft.fftfreq(n, 1/sampling_rate)

    # 진폭 추출 및 양수 구간 한정
    amplitude = np.abs(fft_val) / n * 2
    pos_mask = freqs > 0

    # 추출된 특징들
    final_freqs = freqs[pos_mask]
    final_amps = amplitude[pos_mask]

    # 3. 결과 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t[:100], data[:100])
    plt.title("Raw Data (Time Domain)\n[Complex and Noisy Raw Signal]")
    plt.xlabel("Time")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.stem(final_freqs[:200], final_amps[:200], basefmt=" ")
    plt.title("Extracted Features (Frequency Domain)\n[Clear Features Extracted by FFT: 60Hz & 120Hz]")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_fft_features()
