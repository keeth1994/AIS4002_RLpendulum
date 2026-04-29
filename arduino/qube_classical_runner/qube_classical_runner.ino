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
const float HARD_VOLTAGE_LIMIT = 5.0f;
const float MOTOR_VOLTAGE_SIGN = 1.0f;
const float THETA_SIGN = 1.0f;
const float ALPHA_SIGN = 1.0f;
const float ALPHA_OFFSET_RAD = PI_F;
const float CENTER_TRIM_DEG = 0.0f;
const float MOTOR_BIAS_VOLTAGE = 0.0f;
const int RUN_MODE = 0;  // 0=sensors, 1=motor sign, 2=classical controller
const float TEST_VOLTAGE = 0.5f;

// Arm protection.
const float ARM_LIMIT_RAD = 0.5f * PI_F;
const float ARM_LIMIT_BRAKE_GAIN = 5.0f;
const float ARM_LIMIT_DAMPING = 0.25f;

// Example-code physical parameters.
const float EXAMPLE_PENDULUM_MASS = 0.1f;
const float EXAMPLE_PENDULUM_LENGTH = 0.095f;
const float EXAMPLE_PENDULUM_COM = EXAMPLE_PENDULUM_LENGTH * 0.5f;
const float EXAMPLE_PENDULUM_INERTIA =
    EXAMPLE_PENDULUM_MASS * EXAMPLE_PENDULUM_LENGTH * EXAMPLE_PENDULUM_LENGTH / 3.0f;

// Voltage scaling copied from the example code.
const float ARM_MASS = 0.095f;
const float ARM_LENGTH = 0.085f;
const float GRAVITY = 9.81f;
const float MOTOR_RESISTANCE = 8.4f;
const float MOTOR_TORQUE_CONSTANT = 0.042f;
const float ACCEL_TO_VOLTAGE = (MOTOR_RESISTANCE * ARM_LENGTH * ARM_MASS) / MOTOR_TORQUE_CONSTANT;

// Parameters copied directly from EXAMPLE_CODE/inverted_pendulum.py.
const float REFERENCE_ENERGY = 0.015f;
const float ENERGY_GAIN = 50.0f;
const float SWINGUP_ACCEL_LIMIT = 2.5f;
const float BALANCE_RANGE_DEG = 20.0f;
const float BALANCE_SCALE = 0.33f;
const float KP_THETA = 1.0f * BALANCE_SCALE;
const float KD_THETA = 0.125f * BALANCE_SCALE;
const float KP_POS = 0.07f * BALANCE_SCALE;
const float KD_POS = 0.06f * BALANCE_SCALE;
const float BALANCE_VOLTAGE_LIMIT = 5.0f;
const float SWINGUP_VOLTAGE_LIMIT = 24.0f;
const float FILTER_WC = 500.0f / TWO_PI_F;

float previousThetaDeg = 0.0f;
float previousAlphaDeg = 0.0f;
float filteredThetaDotDeg = 0.0f;
float filteredAlphaDotDeg = 0.0f;
unsigned long previousMicros = 0;
unsigned long stepCount = 0;
bool balanceModeActive = false;
bool resetModeActive = false;
unsigned long resetStartMicros = 0;
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

float pendulumEnergy(float angleRad, float angularVelocityRad) {
  const float kinetic = 0.5f * EXAMPLE_PENDULUM_INERTIA * angularVelocityRad * angularVelocityRad;
  const float potential = EXAMPLE_PENDULUM_MASS * GRAVITY * EXAMPLE_PENDULUM_COM * (1.0f - cosf(angleRad));
  return kinetic + potential;
}

float computeSwingupVoltage(float angleDeg, float angularVelocityRad) {
  const float angleRad = degToRad(angleDeg);
  const float energy = pendulumEnergy(angleRad, angularVelocityRad);
  float u = ENERGY_GAIN * (energy - REFERENCE_ENERGY) * (-angularVelocityRad * cosf(angleRad));
  u = constrain(u, -SWINGUP_ACCEL_LIMIT, SWINGUP_ACCEL_LIMIT);
  return constrain(u * ACCEL_TO_VOLTAGE, -SWINGUP_VOLTAGE_LIMIT, SWINGUP_VOLTAGE_LIMIT);
}

float computeBalanceVoltage(float positionDeg, float angleDeg, float posRateDeg, float angleRateDeg) {
  const float uPos = KP_POS * positionDeg + KD_POS * posRateDeg;
  const float uAng = KP_THETA * angleDeg + KD_THETA * angleRateDeg;
  return constrain(uPos + uAng, -BALANCE_VOLTAGE_LIMIT, BALANCE_VOLTAGE_LIMIT);
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

float computeSettleAngleDeg(float angleDeg) {
  return (angleDeg < 0.0f) ? (angleDeg + 360.0f - 2.0f * angleDeg) : (-360.0f + 2.0f * angleDeg);
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
  resetModeActive = false;
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
  } else if (balanceModeActive && fabsf(alphaDeg) >= BALANCE_RANGE_DEG) {
    balanceModeActive = false;
    resetModeActive = true;
    resetStartMicros = now;
  }

  float voltage = 0.0f;
  if (RUN_MODE == 2) {
    if (resetModeActive) {
      const float settleAngleDeg = computeSettleAngleDeg(alphaDeg);
      voltage = computeSwingupVoltage(settleAngleDeg, degToRad(filteredAlphaDotDeg));
      if ((now - resetStartMicros) * 1.0e-6f >= 2.0f) {
        resetModeActive = false;
      }
    } else if (balanceModeActive) {
      voltage = computeBalanceVoltage(thetaDeg, alphaDeg, filteredThetaDotDeg, filteredAlphaDotDeg);
    } else {
      voltage = computeSwingupVoltage(alphaDeg, degToRad(filteredAlphaDotDeg));
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
    } else if (resetModeActive) {
      Serial.print("reset");
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
