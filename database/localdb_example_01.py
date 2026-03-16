import sqlite3
from datetime import datetime

# 1. 데이터베이스 연결 (파일이 없으면 새로 생성함)
# 별도의 서버 설치 없이 factory.db라는 파일 하나로 관리됩니다.
def create_connection():
    conn = sqlite3.connect('factory.db')
    return conn

# 2. 테이블 생성 (부품함의 서랍 만들기)
def setup_database():
    conn = create_connection()
    cursor = conn.cursor()
    
    # 생산 실적을 저장할 테이블 생성
    # 일시, 제품명, 생산수량, 상태(양호/불량)를 칸으로 나눔
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS production_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("데이터베이스 및 테이블 설정 완료.")

# 3. 데이터 기록하기 (사용자 입력 기반)
def insert_production_data():
    print("\n--- [새 생산 데이터 입력] ---")
    product = input("제품명을 입력하세요: ")
    try:
        qty = int(input("생산 수량을 입력하세요(숫자): "))
    except ValueError:
        print("오류: 수량은 숫자만 입력 가능합니다.")
        return
        
    status = input("상태를 입력하세요(예: 양호/불량): ")
    
    conn = create_connection()
    cursor = conn.cursor()
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 데이터를 테이블에 삽입
    cursor.execute('''
        INSERT INTO production_logs (timestamp, product_name, quantity, status)
        VALUES (?, ?, ?, ?)
    ''', (now, product, qty, status))
    
    conn.commit()
    conn.close()
    print(f"✅ 기록 완료: {product} {qty}개 ({status})")

# 4. 데이터 열람하기 (부품함에서 데이터 찾기)
def view_production_logs():
    conn = create_connection()
    cursor = conn.cursor()
    
    print("\n--- [현재 생산 실적 리스트] ---")
    cursor.execute('SELECT * FROM production_logs')
    rows = cursor.fetchall()
    
    if not rows:
        print("저장된 데이터가 없습니다.")
    else:
        for row in rows:
            print(f"ID: {row[0]} | 시간: {row[1]} | 제품: {row[2]} | 수량: {row[3]} | 상태: {row[4]}")
    
    conn.close()

# --- 메인 메뉴 프로그램 ---
def main_menu():
    setup_database()
    
    while True:
        print("\n==========================")
        print("   공장 데이터 관리 시스템")
        print("==========================")
        print("1. 생산 데이터 입력")
        print("2. 생산 실적 조회")
        print("3. 프로그램 종료")
        
        choice = input("\n원하는 메뉴 번호를 선택하세요: ")
        
        if choice == '1':
            insert_production_data()
        elif choice == '2':
            view_production_logs()
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 입력해 주세요.")

if __name__ == "__main__":
    main_menu()
