#include <math.h>

#include "QUBE.hpp"
#include "qube_policy.h"

QUBE qube;

const float PI_F = 3.14159265358979323846f;
const float DEG_TO_RAD_F = PI_F / 180.0f;

// Main hardware knobs. This policy is a balance controller, so start near upright.
const float CONTROL_DT_SECONDS = 1.0f / 300.0f;
const float POLICY_VOLTAGE_GAIN = 1.0f;
const float HARD_VOLTAGE_LIMIT = 5.0f;
const float ARM_LIMIT_RAD = 0.5f * PI_F;
const float ARM_LIMIT_BRAKE_GAIN = 5.0f;
const float ARM_LIMIT_DAMPING = 0.25f;
const float MOTOR_VOLTAGE_SIGN = 1.0f;
const float THETA_SIGN = 1.0f;
const float ALPHA_SIGN = 1.0f;
const float ALPHA_OFFSET_RAD = PI_F;
const float VELOCITY_FILTER_ALPHA = 0.15f;

const unsigned long START_DELAY_MS = 3000;
const int PRINT_EVERY_STEPS = 10;

float previousTheta = 0.0f;
float previousAlpha = PI_F;
float thetaDot = 0.0f;
float alphaDot = 0.0f;
unsigned long previousMicros = 0;
unsigned long stepCount = 0;

float wrapRad(float angle) {
  while (angle > PI_F) angle -= 2.0f * PI_F;
  while (angle < -PI_F) angle += 2.0f * PI_F;
  return angle;
}

float clipFloat(float value, float limit) {
  if (value > limit) return limit;
  if (value < -limit) return -limit;
  return value;
}

float readThetaRad() {
  return THETA_SIGN * qube.getMotorAngle(false) * DEG_TO_RAD_F;
}

float readAlphaRad() {
  const float rawPendulum = qube.getPendulumAngle(false) * DEG_TO_RAD_F;
  return wrapRad(ALPHA_SIGN * rawPendulum + ALPHA_OFFSET_RAD);
}

void stopMotor() {
  qube.setMotorVoltage(0.0f);
  qube.update();
}

void setup() {
  Serial.begin(115200);
  delay(500);

  qube.begin();
  stopMotor();
  qube.setRGB(0, 0, 999);
  qube.update();

  Serial.println("Pure RL QUBE runner");
  Serial.println("This exported policy is balance-only.");
  Serial.println("Start with the pendulum close to upright and the arm near center.");
  Serial.println("Starting in 3 seconds.");
  delay(START_DELAY_MS);

  qube.resetMotorEncoder();
  qube.resetPendulumEncoder();
  qube.update();

  previousTheta = readThetaRad();
  previousAlpha = readAlphaRad();
  previousMicros = micros();

  qube.setRGB(0, 999, 0);
  qube.update();
}

void loop() {
  qube.update();

  if (qube.hasAmplifierFault() || qube.hasStallError()) {
    stopMotor();
    qube.setRGB(999, 0, 0);
    qube.update();
    Serial.println("QUBE fault detected. Motor stopped.");
    while (true) {
      delay(1000);
    }
  }

  const unsigned long now = micros();
  const float dt = (now - previousMicros) * 1.0e-6f;
  if (dt < CONTROL_DT_SECONDS) {
    return;
  }
  previousMicros = now;

  const float theta = readThetaRad();
  const float alpha = readAlphaRad();
  const float rawThetaDot = (theta - previousTheta) / dt;
  const float rawAlphaDot = wrapRad(alpha - previousAlpha) / dt;
  previousTheta = theta;
  previousAlpha = alpha;

  thetaDot += VELOCITY_FILTER_ALPHA * (rawThetaDot - thetaDot);
  alphaDot += VELOCITY_FILTER_ALPHA * (rawAlphaDot - alphaDot);

  float voltage = qubePolicyPredictVoltage(theta, alpha, thetaDot, alphaDot);
  voltage = clipFloat(voltage * POLICY_VOLTAGE_GAIN, HARD_VOLTAGE_LIMIT);

  float motorVoltage = MOTOR_VOLTAGE_SIGN * voltage;
  if (theta > ARM_LIMIT_RAD) {
    const float brakeVoltage = -ARM_LIMIT_BRAKE_GAIN * (theta - ARM_LIMIT_RAD) - ARM_LIMIT_DAMPING * thetaDot;
    if (motorVoltage > brakeVoltage) motorVoltage = brakeVoltage;
  } else if (theta < -ARM_LIMIT_RAD) {
    const float brakeVoltage = -ARM_LIMIT_BRAKE_GAIN * (theta + ARM_LIMIT_RAD) - ARM_LIMIT_DAMPING * thetaDot;
    if (motorVoltage < brakeVoltage) motorVoltage = brakeVoltage;
  }
  motorVoltage = clipFloat(motorVoltage, HARD_VOLTAGE_LIMIT);
  qube.setMotorVoltage(motorVoltage);

  if ((stepCount % PRINT_EVERY_STEPS) == 0) {
    Serial.print("theta=");
    Serial.print(theta * 180.0f / PI_F, 1);
    Serial.print(" deg, alpha=");
    Serial.print(alpha * 180.0f / PI_F, 1);
    Serial.print(" deg, rawPend=");
    Serial.print(qube.getPendulumAngle(false), 1);
    Serial.print(" deg");
    Serial.print(", thetaDot=");
    Serial.print(thetaDot, 3);
    Serial.print(", alphaDot=");
    Serial.print(alphaDot, 3);
    Serial.print(", voltage=");
    Serial.println(motorVoltage, 3);
  }

  stepCount++;
}
