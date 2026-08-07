# =========================================================
# 1. Import
# =========================================================
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None


# =========================================================
# 2. 함수 및 상수 정의
# =========================================================
load_dotenv()

SENSOR_CSV = "factory_sensor_logs.csv"
USE_LOCAL_LLM = False

SYSTEM_PROMPT = """너는 25년 경력의 설비보전 수석 엔지니어이자 예지보전 AI 에이전트이다.
작업자나 관리자가 설비 점검을 요청하면 다음 순서를 반드시 준수하여 업무를 수행하라:

1. 먼저 `get_sensor_log_summary` 도구를 호출하여 대상 설비의 실시간 센서 로그 통계를 조회하라.
2. 수집된 온도, 진동 RMS, 전류 변화 패턴을 종합 분석하여 이상 원인을 추론하라.
   - [이상 추론 가이드]
     a) 온도 급상승 + 진동 RMS 폭증 + 전류 상승: 윤활 구리스 고갈 및 베어링 과열 파손 가능성 매우 높음.
     b) 온도는 정상 + 특정 주파수 진동만 급증: 회전축 불균형(Unbalance) 또는 축 정렬 불량(Misalignment).
     c) 전류만 급증 + 온도 완만한 상승: 가공부 과부하 또는 구동부 이물질 끼임.
3. 원인이 진단되면, 현장 정비 작업자가 지참해야 할 공구, 안전 조치, 작업 순서를 포함한 '마크다운 정비 보고서'를 작성하라.
4. 작성된 보고서는 `save_maintenance_report` 도구를 사용하여 마크다운 파일로 저장하라.
5. 저장 완료 후 작업자에게 최종 답변으로 결과 요약을 보고하라."""


def generate_mock_sensor_csv(filepath=SENSOR_CSV):
    """현장 진동/온도 센서에서 수집된 시계열 로그 파일 생성"""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=5 * i) for i in range(12)][::-1]

    # MOTOR_01: 윤활유 고갈로 진동과 온도가 함께 급증하는 전형적인 고장 패턴
    data_m1 = {
        "timestamp": timestamps,
        "equipment_id": "MOTOR_01",
        "temp_celsius": [45.0, 46.2, 48.0, 52.1, 58.5, 64.0, 71.2, 78.5, 83.0, 86.5, 89.2, 91.0],
        "vibration_rms": [1.1, 1.2, 1.4, 2.1, 3.5, 4.8, 6.2, 7.5, 8.8, 9.4, 10.1, 10.8],
        "current_amp": [12.0, 12.1, 12.2, 13.0, 14.2, 15.1, 15.8, 16.5, 17.0, 17.2, 17.5, 17.8],
    }

    # MOTOR_02: 매우 안정적으로 작동 중인 정상 모터 패턴
    data_m2 = {
        "timestamp": timestamps,
        "equipment_id": "MOTOR_02",
        "temp_celsius": [41.0, 41.2, 41.5, 41.3, 41.8, 42.0, 41.9, 42.1, 42.0, 41.8, 42.2, 42.1],
        "vibration_rms": [0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8],
        "current_amp": [10.5, 10.5, 10.6, 10.5, 10.5, 10.6, 10.5, 10.5, 10.6, 10.5, 10.5, 10.5],
    }

    df = pd.concat([pd.DataFrame(data_m1), pd.DataFrame(data_m2)], ignore_index=True)
    df.to_csv(filepath, index=False)


@tool
def get_sensor_log_summary(equipment_id: str) -> str:
    """
    지정한 설비(equipment_id)의 최근 센서 로그(온도, 진동 RMS, 전류 수치)를
    Pandas로 분석하여 통계적 추이 요약을 반환하는 연장(Tool).
    설비 이상 원인을 진단하고자 할 때 반드시 첫 번째로 사용해야 한다.

    Args:
        equipment_id (str): 설비 식별 번호 (예: 'MOTOR_01', 'MOTOR_02')
    """
    df = pd.read_csv(SENSOR_CSV)
    target = df[df["equipment_id"] == equipment_id.upper()].copy()
    if target.empty:
        return f"[조회 실패] '{equipment_id}' 설비의 센서 로그가 존재하지 않는다."

    target["timestamp"] = pd.to_datetime(target["timestamp"])
    target = target.sort_values("timestamp")

    start, latest = target.iloc[0], target.iloc[-1]
    temp_change = round(latest["temp_celsius"] - start["temp_celsius"], 2)
    vib_change = round(latest["vibration_rms"] - start["vibration_rms"], 2)
    avg_current = round(target["current_amp"].mean(), 2)
    max_current = round(target["current_amp"].max(), 2)

    temp_trend = "급상승" if temp_change > 15 else ("상승" if temp_change > 5 else "안정")
    vib_trend = "폭증" if vib_change > 5 else ("상승" if vib_change > 2 else "안정")

    return (
        f"=== [{equipment_id}] 최근 1시간 센서 로그 분석 결과 ===\n"
        f"1. 최신 측정 수치 (측정시각: {latest['timestamp'].strftime('%H:%M')}):\n"
        f"   - 현재 온도: {latest['temp_celsius']}°C (1시간 전 대비 {temp_change:+.1f}°C, 추세: {temp_trend})\n"
        f"   - 현재 진동(RMS): {latest['vibration_rms']} mm/s (1시간 전 대비 {vib_change:+}, 추세: {vib_trend})\n"
        f"   - 현재 부하전류: {latest['current_amp']} A (평균: {avg_current} A, 최대: {max_current} A)\n"
        f"2. 설비 안전 평가 기준:\n"
        f"   - 정상 범위: 온도 < 60°C, 진동 RMS < 2.5 mm/s, 전류 < 14 A\n"
        f"   - 경고 범위: 온도 60~85°C, 진동 RMS 2.5~6.0 mm/s\n"
        f"   - 위험 범위: 온도 > 85°C 초과 또는 진동 RMS > 6.0 mm/s 초과"
    )


@tool
def save_maintenance_report(equipment_id: str, report_content: str) -> str:
    """
    작성된 예지보전 이상 분석 및 정비 가이드 보고서를 마크다운(.md) 파일로 저장하는 연장(Tool).
    원인 분석이 완료되면 반드시 이 도구를 호출하여 최종 보고서를 파일로 자동 출력해야 한다.

    Args:
        equipment_id (str): 설비 식별 번호 (예: 'MOTOR_01')
        report_content (str): 마크다운 형식으로 작성된 상세 정비 보고서 내용
    """
    filename = f"정비보고서_{equipment_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    return f"[성공] 마크다운 정비 보고서가 저장되었다: {os.path.abspath(filename)}"


def build_llm():
    if USE_LOCAL_LLM:
        if ChatOllama is None:
            raise RuntimeError("로컬 LLM 사용을 위해 `pip install langchain-community` 필요")
        print(">>> [시스템] 보안 로컬 LLM (Ollama Qwen2.5) 연동 완료.")
        return ChatOllama(model="qwen2.5:14b", temperature=0)

    print(">>> [알림] 무료 클라우드 LLM (Groq - llama-3.3) 모드로 가동합니다.")
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
    )


def build_agent():
    tools = [get_sensor_log_summary, save_maintenance_report]
    return create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


# =========================================================
# 3. 실행
# =========================================================
def main():
    print("\n==================================================")
    print("  [시나리오 1] 예지보전 센서 로그 분석 에이전트 가동")
    print("==================================================\n")

    generate_mock_sensor_csv()
    agent = build_agent()

    user_query = (
        "지금 MOTOR_01 센서 상태 점검하고, 이상 있는 것 같으면 "
        "원인 분석해서 정비 보고서 파일로 만들어줘."
    )
    print(f"[작업자 요청]: {user_query}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

    print("\n==================================================")
    print("[AI 수석 엔지니어 최종 답변]:")
    print(result["messages"][-1].content)
    print("==================================================")


if __name__ == "__main__":
    main()
