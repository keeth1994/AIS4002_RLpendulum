// Minimal Arduino sketch skeleton for a QUBE-Servo 2 PPO balance policy.
//
// This sketch will not compile until you replace the placeholder hardware
// functions with the QUBE-Servo 2 Embedded/Arduino API from your lab handout.
// Keep voltage limits low for the first tests and always use an emergency stop.

#include "qube_policy.h"

const float CONTROL_DT_SECONDS = 0.01f;  // 100 Hz, matching the simulator.
const float ARM_LIMIT_RAD = 1.04719755f; // 60 degrees.

float previousTheta = 0.0f;
float previousAlpha = 0.0f;
unsigned long previousMicros = 0;

float readMotorAngleRad() {
  // TODO: Replace with QUBE encoder 0 conversion from counts to radians.
  return 0.0f;
}

float readPendulumAngleRadUprightZero() {
  // TODO: Replace with QUBE encoder 1 conversion.
  // Must match simulator convention: alpha = 0 upright, alpha = pi downward.
  return 0.0f;
}

void writeMotorVoltage(float voltage) {
  // TODO: Replace with QUBE motor voltage command.
  // First tests should use +/-1 V or +/-2 V limits.
}

void enableMotor(bool enabled) {
  // TODO: Replace with QUBE motor enable/disable call.
}

void setup() {
  enableMotor(false);
  previousTheta = readMotorAngleRad();
  previousAlpha = readPendulumAngleRadUprightZero();
  previousMicros = micros();
  enableMotor(true);
}

void loop() {
  unsigned long now = micros();
  float dt = (now - previousMicros) * 1.0e-6f;
  if (dt < CONTROL_DT_SECONDS) {
    return;
  }
  previousMicros = now;

  float theta = readMotorAngleRad();
  float alpha = readPendulumAngleRadUprightZero();
  float thetaDot = (theta - previousTheta) / dt;
  float alphaDot = (alpha - previousAlpha) / dt;
  previousTheta = theta;
  previousAlpha = alpha;

  if (fabs(theta) > ARM_LIMIT_RAD) {
    writeMotorVoltage(0.0f);
    enableMotor(false);
    while (true) {
      delay(100);
    }
  }

  float voltage = qubePolicyPredictVoltage(theta, alpha, thetaDot, alphaDot);
  writeMotorVoltage(voltage);
}
