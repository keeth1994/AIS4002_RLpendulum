// Low-voltage motor sign check for QUBE.
// Upload only with the pendulum removed or held safely. This applies small
// alternating voltages so you can verify motor direction and encoder sign.

#include "QUBE.hpp"

const float QUBE_DEG_TO_RAD = 0.01745329252f;
const float TEST_VOLTAGE = 0.5f;

QUBE qube(10);

void setup() {
  Serial.begin(115200);
  qube.begin();
  qube.resetMotorEncoder();
  qube.setRGB(500, 500, 0);
  qube.update();
}

void loop() {
  qube.setMotorVoltage(TEST_VOLTAGE);
  qube.update();
  delay(400);
  printState("positive");

  qube.setMotorVoltage(0.0f);
  qube.update();
  delay(600);

  qube.setMotorVoltage(-TEST_VOLTAGE);
  qube.update();
  delay(400);
  printState("negative");

  qube.setMotorVoltage(0.0f);
  qube.update();
  delay(1200);
}

void printState(const char* phase) {
  float theta = qube.getMotorAngle(false) * QUBE_DEG_TO_RAD;
  Serial.print(phase);
  Serial.print(" theta_rad=");
  Serial.print(theta);
  Serial.print(" motor_deg=");
  Serial.print(qube.getMotorAngle(false));
  Serial.print(" voltage=");
  Serial.print(qube.getMotorVoltage());
  Serial.print(" ampFault=");
  Serial.print(qube.hasAmplifierFault());
  Serial.print(" stall=");
  Serial.print(qube.hasStallDetected());
  Serial.print(" stallError=");
  Serial.println(qube.hasStallError());
}
