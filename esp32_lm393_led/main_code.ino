#include <WiFi.h>
#include <PubSubClient.h>

// 1. 우리 공장 환경에 맞게 수정할 부분
const char* ssid = "YOUR_WIFI_SSID";          // 와이파이 이름
const char* password = "YOUR_WIFI_PASSWORD";  // 와이파이 비밀번호
const char* mqtt_server = "YOUR_SERVER_IP";   // MQTT 서버(브로커) 주소

// 2. 핀 번호 설정 (앞에서 약속한 대로)
#define lightPin 14  // 광센서 연결 (D14)
#define ledPin 12    // 상태 표시용 LED 연결 (D12)

WiFiClient espClient;
PubSubClient client(espClient);

long stTime;
int ndx = 0;
int onTime = 60;

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
}

void reconnect() {
  // 서버와 연결이 끊어졌을 때 다시 연결 시도
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32_Factory_Sensor")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(lightPin, INPUT);
  pinMode(ledPin, OUTPUT);
  
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  
  stTime = millis();
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  ndx++;

  // --- 기존 가동 시간 측정 로직 ---
  int opLevel = digitalRead(lightPin);  // 0(켜짐), 1(꺼짐) 읽기
  digitalWrite(ledPin, !opLevel);       // 센서가 0일 때 LED를 켜기 위해 반전(!)
  onTime -= opLevel;                    // 꺼진 시간만큼 차감하여 가동 시간 계산
  
  // 정확히 1초 간격 유지
  while (millis() - stTime < ndx * 1000) {
    delay(10);
  }

  // 60초(1분)가 되면 서버로 전송
  if (ndx >= 60) {
    char msg[10];
    snprintf(msg, 10, "%d", onTime);
    
    Serial.print("Sending 1min Uptime: ");
    Serial.println(msg);
    
    // 'factory/machine1/uptime' 이라는 주제(Topic)로 데이터 전송
    client.publish("factory/machine1/uptime", msg);

    // 다음 측정을 위해 초기화
    stTime = millis();
    onTime = 60;
    ndx = 0;
  }
}
