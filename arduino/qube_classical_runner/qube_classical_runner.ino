// Classical hybrid swing-up + PD balance controller for QUBE-Servo 2.
// Start with the pendulum hanging down and the arm near center.

#include <math.h>

#include "QUBE.hpp"

QUBE qube;

const float PI_F = 3.14159265358979323846f;
const float TWO_PI_F = 2.0f * PI_F;
const float DEG_TO_RAD_F = PI_F / 180.0f;

// Control timing.
const float CONTROL_DT_SECONDS = 1.0f / 300.0f;
const unsigned long START_DELAY_MS = 3000;
const int PRINT_EVERY_STEPS = 10;

// Hardware-facing calibration.
const float HARD_VOLTAGE_LIMIT = 8.0f;
const float MOTOR_VOLTAGE_SIGN = 1.0f;
const float THETA_SIGN = 1.0f;
const float ALPHA_SIGN = 1.0f;
const float ALPHA_OFFSET_RAD = PI_F;
const float CENTER_TRIM_DEG = 0.0f;
const float MOTOR_BIAS_VOLTAGE = 0.0f;
const int RUN_MODE = 2;  // 0=sensors, 1=motor sign, 2=classical controller
const float TEST_VOLTAGE = 0.5f;

// Arm protection.
const float ARM_LIMIT_RAD = 0.5f * PI_F;
const float ARM_LIMIT_BRAKE_GAIN = 5.0f;
const float ARM_LIMIT_DAMPING = 0.25f;

// Hybrid controller settings aligned with the simulator baseline.
const float BALANCE_RANGE_DEG = 20.0f;
const float BALANCE_EXIT_RANGE_DEG = 35.0f;
const float SWINGUP_FREQUENCY_HZ = 1.5f;
const float SWINGUP_AMPLITUDE = 8.0f;
const float THETA_GAIN = -1.3235294117647058f;
const float ALPHA_GAIN = 18.90760723931717f;
const float THETA_DOT_GAIN = -1.134453781512605f;
const float ALPHA_DOT_GAIN = 2.3634509049146462f;
const float ARM_CENTERING_GAIN = 1.3235294117647058f;
const float ARM_CENTERING_RATE_GAIN = 1.134453781512605f;
const float BALANCE_VOLTAGE_LIMIT = 5.0f;
const float FILTER_WC = 500.0f / TWO_PI_F;

float previousThetaDeg = 0.0f;
float previousAlphaDeg = 0.0f;
float filteredThetaDotDeg = 0.0f;
float filteredAlphaDotDeg = 0.0f;
unsigned long previousMicros = 0;
unsigned long stepCount = 0;
bool balanceModeActive = false;
bool testVoltagePositive = true;
unsigned long lastTestToggleMicros = 0;

float wrapRad(float angle) {
  while (angle > PI_F) angle -= TWO_PI_F;
  while (angle < -PI_F) angle += TWO_PI_F;
  return angle;
}

float clipFloat(float value, float limit) {
  if (value > limit) return limit;
  if (value < -limit) return -limit;
  return value;
}

float readThetaDeg() {
  return THETA_SIGN * qube.getMotorAngle() - CENTER_TRIM_DEG;
}

float wrapDeg180(float angleDeg) {
  while (angleDeg > 180.0f) angleDeg -= 360.0f;
  while (angleDeg < -180.0f) angleDeg += 360.0f;
  return angleDeg;
}

float readAlphaDegForExample() {
  const float rawPendulumDeg = ALPHA_SIGN * qube.getPendulumAngle();
  const float wrapped = fmodf(rawPendulumDeg, 360.0f);
  const float angleDeg = (wrapped > 0.0f) ? (-180.0f + wrapped) : (180.0f + wrapped);
  return wrapDeg180(angleDeg);
}

float degToRad(float angleDeg) {
  return angleDeg * DEG_TO_RAD_F;
}

float computeSwingupVoltage(float thetaRad, float thetaDotRad, float timeS) {
  float voltage = SWINGUP_AMPLITUDE * sinf(2.0f * PI_F * SWINGUP_FREQUENCY_HZ * timeS);
  voltage -= ARM_CENTERING_GAIN * thetaRad + ARM_CENTERING_RATE_GAIN * thetaDotRad;
  return constrain(voltage, -SWINGUP_AMPLITUDE, SWINGUP_AMPLITUDE);
}

float computeBalanceVoltage(float thetaRad, float alphaRad, float thetaDotRad, float alphaDotRad) {
  float voltage = -(
      THETA_GAIN * thetaRad
      + ALPHA_GAIN * alphaRad
      + THETA_DOT_GAIN * thetaDotRad
      + ALPHA_DOT_GAIN * alphaDotRad);
  return constrain(voltage, -BALANCE_VOLTAGE_LIMIT, BALANCE_VOLTAGE_LIMIT);
}

float applyArmLimitProtection(float thetaRad, float thetaDotRad, float voltage) {
  float protectedVoltage = voltage;
  if (thetaRad > ARM_LIMIT_RAD) {
    const float brakeVoltage = -ARM_LIMIT_BRAKE_GAIN * (thetaRad - ARM_LIMIT_RAD) - ARM_LIMIT_DAMPING * thetaDotRad;
    if (protectedVoltage > brakeVoltage) protectedVoltage = brakeVoltage;
  } else if (thetaRad < -ARM_LIMIT_RAD) {
    const float brakeVoltage = -ARM_LIMIT_BRAKE_GAIN * (thetaRad + ARM_LIMIT_RAD) - ARM_LIMIT_DAMPING * thetaDotRad;
    if (protectedVoltage < brakeVoltage) protectedVoltage = brakeVoltage;
  }
  return clipFloat(protectedVoltage, HARD_VOLTAGE_LIMIT);
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

  Serial.println("QUBE classical controller");
  if (RUN_MODE == 0) {
    Serial.println("RUN_MODE=0 sensor check");
    Serial.println("Start with the pendulum hanging down and verify alpha is near pi or -pi.");
  } else if (RUN_MODE == 1) {
    Serial.println("RUN_MODE=1 motor sign check");
    Serial.println("Hold the pendulum safely and verify +/-0.5 V moves theta in the expected direction.");
  } else {
    Serial.println("RUN_MODE=2 classical controller");
    Serial.println("Start with the pendulum hanging down and the arm near center.");
  }
  Serial.println("Starting in 3 seconds.");
  delay(START_DELAY_MS);

  qube.resetMotorEncoder();
  qube.resetPendulumEncoder();
  qube.update();

  previousThetaDeg = readThetaDeg();
  previousAlphaDeg = readAlphaDegForExample();
  previousMicros = micros();
  balanceModeActive = false;
  filteredThetaDotDeg = 0.0f;
  filteredAlphaDotDeg = 0.0f;
  testVoltagePositive = true;
  lastTestToggleMicros = micros();

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

  const float thetaDeg = readThetaDeg();
  const float alphaDeg = readAlphaDegForExample();
  const float rawThetaDotDeg = (thetaDeg - previousThetaDeg) / dt;
  const float rawAlphaDotDeg = wrapDeg180(alphaDeg - previousAlphaDeg) / dt;
  previousThetaDeg = thetaDeg;
  previousAlphaDeg = alphaDeg;

  filteredThetaDotDeg += FILTER_WC * dt * (rawThetaDotDeg - filteredThetaDotDeg);
  filteredAlphaDotDeg += FILTER_WC * dt * (rawAlphaDotDeg - filteredAlphaDotDeg);

  if (RUN_MODE == 0) {
    qube.setMotorVoltage(0.0f);
  } else if (RUN_MODE == 1) {
    if ((now - lastTestToggleMicros) * 1.0e-6f >= 1.0f) {
      testVoltagePositive = !testVoltagePositive;
      lastTestToggleMicros = now;
    }
    qube.setMotorVoltage(MOTOR_VOLTAGE_SIGN * (testVoltagePositive ? TEST_VOLTAGE : -TEST_VOLTAGE));
  } else if (!balanceModeActive && fabsf(alphaDeg) < BALANCE_RANGE_DEG) {
    balanceModeActive = true;
  } else if (balanceModeActive && fabsf(alphaDeg) > BALANCE_EXIT_RANGE_DEG) {
    balanceModeActive = false;
  }

  float voltage = 0.0f;
  if (RUN_MODE == 2) {
    if (balanceModeActive) {
      voltage = computeBalanceVoltage(
          degToRad(thetaDeg),
          degToRad(alphaDeg),
          degToRad(filteredThetaDotDeg),
          degToRad(filteredAlphaDotDeg));
    } else {
      voltage = computeSwingupVoltage(
          degToRad(thetaDeg),
          degToRad(filteredThetaDotDeg),
          stepCount * CONTROL_DT_SECONDS);
    }

    voltage += MOTOR_BIAS_VOLTAGE;
    voltage = applyArmLimitProtection(degToRad(thetaDeg), degToRad(filteredThetaDotDeg), voltage);
    voltage = clipFloat(voltage, HARD_VOLTAGE_LIMIT);

    const float motorVoltage = MOTOR_VOLTAGE_SIGN * voltage;
    qube.setMotorVoltage(motorVoltage);
  } else {
    voltage = qube.getMotorVoltage();
  }

  if ((stepCount % PRINT_EVERY_STEPS) == 0) {
    Serial.print("mode=");
    if (RUN_MODE == 0) {
      Serial.print("sensors");
    } else if (RUN_MODE == 1) {
      Serial.print(testVoltagePositive ? "test_pos" : "test_neg");
    } else {
      Serial.print(balanceModeActive ? "balance" : "swingup");
    }
    const float thetaRad = degToRad(thetaDeg);
    const float alphaRad = degToRad(alphaDeg);
    const float thetaDotRad = degToRad(filteredThetaDotDeg);
    const float alphaDotRad = degToRad(filteredAlphaDotDeg);
    Serial.print(" theta=");
    Serial.print(thetaRad, 4);
    Serial.print(" alpha=");
    Serial.print(alphaRad, 4);
    Serial.print(" thetaDot=");
    Serial.print(thetaDotRad, 4);
    Serial.print(" alphaDot=");
    Serial.print(alphaDotRad, 4);
    Serial.print(" voltage=");
    Serial.print(qube.getMotorVoltage(), 4);
    Serial.print(" ampFault=");
    Serial.print(qube.hasAmplifierFault());
    Serial.print(" stall=");
    Serial.print(qube.hasStallDetected());
    Serial.print(" stallError=");
    Serial.println(qube.hasStallError());
  }

  stepCount++;
}
