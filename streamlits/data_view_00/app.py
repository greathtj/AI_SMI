import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 사이드바 설정 (사용자 입력)
st.sidebar.title("⚙️ 분석 조건 설정")
data_count = st.sidebar.slider("데이터 샘플 수", 10, 50, 20)
noise_level = st.sidebar.selectbox("변동성 수준", ["낮음", "중간", "높음"])

noise_map = {"낮음": 1, "중간": 2, "높음": 4}
noise = noise_map[noise_level]

# 2. 데이터 생성 로직
chart_data = pd.DataFrame({
    '시간': [f"{i}:00" for i in range(data_count)],
    '온도': 70 + np.sin(np.arange(data_count) / 3) * 5 + np.random.randn(data_count) * noise,
    '압력': 15 + np.random.randn(data_count) * 2
})

# 3. 메인 화면 구성
st.title("🔥 Streamlit 실무 분석 앱")

col1, col2, col3 = st.columns(3)
col1.metric("평균 온도", f"{chart_data['온도'].mean():.1f}°C")
col2.metric("최고 압력", f"{chart_data['압력'].max():.1f} bar")
col3.success("정상 작동" if chart_data['온도'].max() < 78 else "주의 요함")

# 4. 차트 출력
tab1, tab2 = st.tabs(["라인 차트", "데이터 테이블"])
with tab1:
    fig = px.line(chart_data, x='시간', y=['온도', '압력'])
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.dataframe(chart_data)
