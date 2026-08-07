import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()  # 프로젝트 폴더의 .env 파일에서 API 키를 읽어온다
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

# ---------------------------------------------------------
# 1. 모델 선택 (로컬 Ollama / 무료 클라우드 / 상용 OpenAI)
# ---------------------------------------------------------
# USE_LOCAL_LLM = True 로 설정하면 3~4장에서 세팅한 폐쇄형 로컬 Ollama를 사용한다.
USE_LOCAL_LLM = False

# 무료 클라우드 LLM 프로바이더 선택 (USE_LOCAL_LLM=False 일 때만 적용)
#   "groq"   : Groq 무료 티어 (Llama 3.3) - OpenAI 호환이라 코드가 간단함
#   "gemini" : Google Gemini Flash 무료 티어 - 품질·속도 우수
#   "openrouter": OpenRouter 무료(:free) 모델
#   "openai" : 상용 OpenAI (유료, api_key 필요)
CLOUD_PROVIDER = "groq"

if USE_LOCAL_LLM:
    # 3장 & 4.4절에서 추천한 로컬 모델 (Qwen 2.5 또는 Llama 3.1)
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="qwen2.5:14b", temperature=0)
    print(">>> [알림] 보안 로컬 LLM (Ollama - Qwen2.5) 모드로 가동합니다.")
elif CLOUD_PROVIDER == "groq":
    # Groq 무료 티어 (Llama 3.3) - API 키 필요: https://console.groq.com/keys
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
    )
    print(">>> [알림] 무료 클라우드 LLM (Groq - Llama 3.3) 모드로 가동합니다.")
elif CLOUD_PROVIDER == "gemini":
    # Google Gemini Flash 무료 티어 - API 키: https://aistudio.google.com/apikey
    # 설치: pip install langchain-google-genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )
    print(">>> [알림] 무료 클라우드 LLM (Google Gemini Flash) 모드로 가동합니다.")
elif CLOUD_PROVIDER == "openrouter":
    # OpenRouter 무료 모델 - API 키: https://openrouter.ai/keys
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    )
    print(">>> [알림] 무료 클라우드 LLM (OpenRouter) 모드로 가동합니다.")
else:
    # 상용 OpenAI API 모드 (유료)
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.environ.get("OPENAI_API_KEY", ""))
    print(">>> [알림] 상용 OpenAI API 모드로 가동합니다.")

# ---------------------------------------------------------
# 2. 에이전트가 사용할 연장(Tools) 정의
# ---------------------------------------------------------

# 도구 1: 외부 인터넷 정보 검색 연장
web_search_tool = DuckDuckGoSearchRun()

# 도구 2: 내부 공장 설비 상태 조회 연장 (2장 레거시 DB 연동부)
@tool
def get_factory_equipment_status(target: str) -> str:
    """
    공장 내부 설비나 라인의 실시간 상태를 조회하는 도구.
    입력값(target)으로 '라인1', '라인2', '모터01', '절삭유' 등을 받을 수 있다.
    """
    factory_db = {
        "라인1": "정상 가동 중 (생산 목표 달성률 94%)",
        "라인2": "비상 정지 중 (사유: 컨베이어 벨트 과부하 센서 작동)",
        "모터01": "현재 온도 88.5°C (과열 주의보 발령 중)",
        "절삭유": "1번 가공기 절삭유 농도 4.2% (기준치: 5%~8% 대비 부족, 보충 필요)"
    }
    
    for key, value in factory_db.items():
        if key in target:
            return f"[현장 DB 조회 결과] {key}: {value}"
            
    return f"[현장 DB 조회 결과] '{target}'에 대한 실시간 수치를 찾을 수 없다."

# 에이전트에게 부여할 도구 목록
tools = [web_search_tool, get_factory_equipment_status]

# ---------------------------------------------------------
# 3. 시스템 페르소나(Prompt) 설정
# ---------------------------------------------------------
system_prompt = """너는 20년 경력의 베테랑 스마트 공장 AI 전문가이다.
현장 작업자나 CEO의 질문을 받으면 다음 원칙에 따라 답변하라:
1. 공장의 설비 상태 주거나 실시간 데이터 조회가 필요하면 `get_factory_equipment_status` 도구를 사용하라.
2. 외부 기술 표준, 매뉴얼, 일반적인 공학 지식이 필요하면 `duckduckgo_search` 도구를 사용하라.
3. 두 가지가 모두 필요하다면 두 도구를 순차적으로 모두 사용하라.
4. 답변은 현장에서 바로 행동할 수 있도록 결론부터 쉽고 명확하게 작성하라."""

# ---------------------------------------------------------
# 4. 에이전트 및 실행기 조립
# ---------------------------------------------------------
agent = create_agent(llm, tools, system_prompt=system_prompt, debug=True)

# ---------------------------------------------------------
# 5. 사용자 입력을 받는 대화형 루프 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n==================================================")
    print("  스마트 공장 현장 지원 AI 에이전트 가동 시작")
    print("  (프로그램을 종료하려면 'q' 또는 '종료'를 입력하라)")
    print("==================================================\n")

    chat_history = []

    while True:
        user_input = input("\n[작업자/CEO 입력]: ")

        # 종료 조건 체크
        if user_input.lower().strip() in ['q', 'quit', 'exit', '종료']:
            print("에이전트 가동을 중지한다.")
            break

        if not user_input.strip():
            continue

        try:
            # 에이전트에게 질문 전달 및 실행
            result = agent.invoke({
                "messages": chat_history + [("human", user_input)]
            })

            chat_history.append(("human", user_input))
            chat_history.append(("ai", result["messages"][-1].content))

            print("\n" + "="*40)
            print(f"[AI 수석 엔지니어 답변]:\n{result['messages'][-1].content}")
            print("="*40)
            
        except Exception as e:
            print(f"\n[오류 발생]: 처리 중 문제가 발생했다 - {e}")
