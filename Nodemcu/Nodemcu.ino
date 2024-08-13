#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Wire.h>
#include <ADS1X15.h>

const char* ssid = "IUNC-Air";
const char* password = "iqra$4321";

ESP8266WebServer server(100);

// Initialize two ADS1115 instances
ADS1115 ads1(0x48); // Address 0x48
ADS1115 ads2(0x49); // Address 0x49

void setup() {
  Serial.begin(9600);
  delay(100);

  // Initialize Wi-Fi connection
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  // Print IP address after successful connection
  IPAddress ip = WiFi.localIP();
  Serial.print("Connected to WiFi. IP address: ");
  Serial.println(ip);

  Wire.begin();
  // Initialize ADS1115
  ads1.begin();
  ads2.begin();

  Serial.println("Connected to WiFi");

  // Setup web server routes
  server.on("/", HTTP_GET, []() {
    String json = "{";
    json += "\"ADS1115_0x48_A0\":" + String(ads1.readADC(0)) + ",";
    json += "\"ADS1115_0x48_A1\":" + String(ads1.readADC(1)) + ",";
    json += "\"ADS1115_0x48_A3\":" + String(ads1.readADC(2)) + ",";
    json += "\"ADS1115_0x49_A0\":" + String(ads2.readADC(0)) + ",";
    json += "\"ADS1115_0x49_A1\":" + String(ads2.readADC(1));
    json += "}";
    server.send(200, "application/json", json);
  });

  // Start the web server
  server.begin();
}

void loop() {
  // Handle client requests
  server.handleClient();
}
