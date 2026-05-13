// Sensor-only QUBE check. Upload this before the RL policy runner.
// It prints motor and pendulum angles without commanding motor voltage.

#include "QUBE.hpp"

const float QUBE_DEG_TO_RAD = 0.01745329252f;

QUBE qube(10);

void setup() {
  Serial.begin(115200);
  qube.begin();
  qube.setMotorVoltage(0.0f);
  qube.setRGB(0, 0, 500);
  qube.resetMotorEncoder();

  // Start with the pendulum hanging down. The reported alpha maps bottom to pi.
  qube.resetPendulumEncoder();
  qube.update();
}

float wrapAngle(float angle) {
  while (angle > 3.14159265f) angle -= 6.28318531f;
  while (angle < -3.14159265f) angle += 6.28318531f;
  return angle;
}

void loop() {
  qube.setMotorVoltage(0.0f);
  qube.update();

  long motorCount = qube.getMotorEncoder();
  long pendulumCount = qube.getPendulumEncoder();
  float theta = qube.getMotorAngle(false) * QUBE_DEG_TO_RAD;
  float rawFromDown = qube.getPendulumAngle(false) * QUBE_DEG_TO_RAD;
  float alpha = wrapAngle(rawFromDown + 3.14159265f);

  Serial.print("raw=[");
  for (int i = 0; i < 6; ++i) {
    if (i > 0) Serial.print(",");
    Serial.print((long)qube.input[i]);
  }
  Serial.print("] ");
  Serial.print("motor_count=");
  Serial.print(motorCount);
  Serial.print(" pendulum_count=");
  Serial.print(pendulumCount);
  Serial.print(" theta_rad=");
  Serial.print(theta);
  Serial.print(" alpha_rad=");
  Serial.print(alpha);
  Serial.print(" motor_deg=");
  Serial.print(qube.getMotorAngle(false));
  Serial.print(" pendulum_deg=");
  Serial.println(qube.getPendulumAngle(false));

  delay(100);
}
