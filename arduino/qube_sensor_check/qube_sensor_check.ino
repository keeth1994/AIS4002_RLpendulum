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

  float theta = qube.getMotorAngle(false) * QUBE_DEG_TO_RAD;
  float rawFromDown = qube.getPendulumAngle(false) * QUBE_DEG_TO_RAD;
  float alpha = wrapAngle(rawFromDown + 3.14159265f);

  Serial.print("theta_rad=");
  Serial.print(theta);
  Serial.print(" alpha_rad=");
  Serial.print(alpha);
  Serial.print(" motor_deg=");
  Serial.print(qube.getMotorAngle(false));
  Serial.print(" pendulum_deg=");
  Serial.println(qube.getPendulumAngle(false));

  delay(100);
}
