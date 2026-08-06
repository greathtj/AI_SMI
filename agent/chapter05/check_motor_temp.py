from langchain_core.tools import tool

@tool
def check_motor_temperature(motor_id: str) -> str:
    """
    지정한 모터(motor_id)의 현재 온도와 가동 상태를 현장 DB(SCADA/MES)에서 조회하는 도구.
    모터 과열 여부나 이상 징후를 진단할 때 반드시 이 도구를 사용해야 한다.
    
    Args:
        motor_id (str): 모터 식별 번호 (예: 'MOTOR_01', 'MOTOR_02')
    """
    # 실제 현장에서는 2장에서 언급한 SCADA, MES, InfluxDB 등 DB와 연동된다.
    # 이해를 돕기 위해 가상의 DB 데이터를 사용한다.
    mock_sensor_db = {
        "MOTOR_01": {"temp": 42.5, "vibration": "정상", "status": "가동 중"},
        "MOTOR_02": {"temp": 88.5, "vibration": "심함", "status": "과열 경고"},
        "MOTOR_03": {"temp": 18.0, "vibration": "없음", "status": "정지"},
    }
    
    info = mock_sensor_db.get(motor_id.upper())
    
    if info:
        return (
            f"[{motor_id}] 실시간 현장 데이터 수집 결과:\n"
            f"- 현재 온도: {info['temp']}°C\n"
            f"- 진동 상태: {info['vibration']}\n"
            f"- 가동 현황: {info['status']}"
        )
    else:
        return f"[{motor_id}] 해당 모터 번호는 현장 센서 DB에 존재하지 않는다."
