// QUBE-Servo 2 swing-up policy runner.
//
// Keep voltage limits low for the first tests and always use an emergency stop.

#include "qube_policy.h"
#include "QUBE.hpp"

// Change this number before uploading to choose what the sketch does:
// 0 = sensor print only, no motor voltage
// 1 = low-voltage motor sign test
// 2 = classical swing-up + balance
// 3 = RL swing-up policy. Use only after exporting a policy trained from down.
// 4 = RL policy dry-run telemetry, no motor voltage
// 5 = policy edge/center diagnostic, no motor voltage
// 6 = open-loop swing power test, no policy
// 7 = arm centering sign test, no pendulum swing-up
const int RUN_MODE = 3;

const float CONTROL_DT_SECONDS = 0.01f;           // 100 Hz, matching the simulator.
const float ARM_LIMIT_RAD = 1.04719755f;          // 60 degrees, balance-only reference.
const float SWINGUP_ARM_LIMIT_RAD = 1.57079633f;  // 90 degrees, physical swing-up range.
const float SOFT_LIMIT_RESUME_RAD = 1.39626340f;  // 80 degrees, hysteresis before retry.
const float RUNNER_VOLTAGE_LIMIT = 4.0f;          // Hardware cap after policy/swing-up scaling.
const float RL_POLICY_FORCE_GAIN = 1.5f;          // 1.5 * 2 V exported policy gives up to 3 V.
const float RL_POLICY_MIN_ABS_VOLTAGE = 0.75f;    // Helps overcome motor deadband near hanging-down start.
const float MODE3_STARTUP_TEST_SECONDS = 0.0f;    // Set >0 only when debugging whether mode 3 drives the motor.
const float MODE3_STARTUP_TEST_VOLTAGE = 1.0f;
const bool USE_SWINGUP_ASSIST_IN_MODE3 = true;
const bool USE_CLASSICAL_BALANCE_IN_MODE3 = true;
const float BALANCE_ENTER_ALPHA_RAD = 0.80f;      // Enter balance within ~46 degrees of upright.
const float BALANCE_EXIT_ALPHA_RAD = 1.10f;       // Resume swing-up outside ~63 degrees.
const float RL_HANDOVER_ALPHA_RAD = 0.65f;        // Use RL within ~37 degrees if classical balance is disabled.
const float SWINGUP_VOLTAGE_LIMIT = 3.5f;
const unsigned long SWINGUP_PULSE_HALF_PERIOD_MS = 550;
const float SWINGUP_CENTERING_THETA_GAIN = 0.18f;
const float SWINGUP_CENTERING_THETA_DOT_GAIN = 0.03f;
const float OPEN_LOOP_TEST_VOLTAGE = 3.0f;
const float OPEN_LOOP_TEST_FREQUENCY_HZ = 1.8f;
const float OPEN_LOOP_PUMP_REGION_RAD = 0.78539816f;    // 45 degrees.
const float OPEN_LOOP_CENTER_REGION_RAD = 1.04719755f;  // 60 degrees.
const float BALANCE_SWITCH_RAD = 0.35f;  // About 20 degrees from upright.
const float VELOCITY_FILTER_ALPHA = 0.15f;
const float QUBE_DEG_TO_RAD = 0.01745329252f;
const bool ENABLE_EDGE_CENTERING_ASSIST = false;
const float EDGE_ASSIST_START_RAD = 1.22173048f;  // 70 degrees.
const float EDGE_ASSIST_THETA_GAIN = 1.5f;
const float EDGE_ASSIST_THETA_DOT_GAIN = 0.15f;
const float CENTER_TEST_VOLTAGE_LIMIT = 0.35f;
const float CENTER_TEST_THETA_GAIN = 0.55f;
const float CENTER_TEST_THETA_DOT_GAIN = 0.06f;
const float CENTER_TEST_DEADBAND_RAD = 0.04f;
const float CENTER_TRIM_DEG = -20.0f;   // Hardware trim found with RUN_MODE 7; keeps physical center at 0.
const float CENTER_TRIM_RAD = CENTER_TRIM_DEG * QUBE_DEG_TO_RAD;
const float MOTOR_BIAS_VOLTAGE = 0.0f;  // Positive adds a small constant push in powered modes.

// Adjust these if sensor mode shows that signs/zeros differ from simulation.
const float THETA_SIGN = 1.0f;
const float ALPHA_SIGN = 1.0f;
const float MOTOR_VOLTAGE_SIGN = 1.0f;
const float THETA_OFFSET_RAD = 0.0f;
const float ALPHA_OFFSET_RAD = 0.0f;

QUBE qube(10);

float previousTheta = 0.0f;
float previousAlpha = 0.0f;
float filteredThetaDot = 0.0f;
float filteredAlphaDot = 0.0f;
unsigned long previousMicros = 0;
bool controllerArmed = false;
bool softLimitActive = false;
bool balanceModeActive = false;
unsigned long controlStartMillis = 0;

void visualBootPattern() {
  qube.setRGB(999, 0, 0);
  qube.update();
  delay(250);
  qube.setRGB(0, 999, 0);
  qube.update();
  delay(250);
  qube.setRGB(0, 0, 999);
  qube.update();
  delay(250);
  qube.setRGB(0, 0, 0);
  qube.update();
}

float readMotorAngleRad() {
  return THETA_SIGN * (qube.getMotorAngle(false) * QUBE_DEG_TO_RAD - THETA_OFFSET_RAD);
}

float readPendulumAngleRadUprightZero() {
  // Reset the pendulum encoder while it hangs down. The policy still receives
  // alpha=0 at the top and alpha=pi at the bottom.
  float rawFromDown = qube.getPendulumAngle(false) * QUBE_DEG_TO_RAD;
  float alpha = rawFromDown + 3.14159265f - ALPHA_OFFSET_RAD;
  while (alpha > 3.14159265f) alpha -= 6.28318531f;
  while (alpha < -3.14159265f) alpha += 6.28318531f;
  return ALPHA_SIGN * alpha;
}

float wrapRad(float angle) {
  while (angle > 3.14159265f) angle -= 6.28318531f;
  while (angle < -3.14159265f) angle += 6.28318531f;
  return angle;
}

void writeMotorVoltage(float voltage) {
  if (fabs(voltage) > 0.001f) {
    voltage += MOTOR_BIAS_VOLTAGE;
  }
  qube.setMotorVoltage(MOTOR_VOLTAGE_SIGN * voltage);
  qube.update();
}

void showVoltageDirection(float voltage) {
  if (voltage > 0.05f) {
    qube.setRGB(0, 700, 0);
  } else if (voltage < -0.05f) {
    qube.setRGB(0, 0, 700);
  } else {
    qube.setRGB(700, 700, 700);
  }
}

void showBalanceMode(float voltage) {
  if (voltage > 0.05f) {
    qube.setRGB(700, 250, 0);
  } else if (voltage < -0.05f) {
    qube.setRGB(700, 0, 700);
  } else {
    qube.setRGB(700, 700, 0);
  }
}

void enableMotor(bool enabled) {
  if (!enabled) {
    qube.setMotorVoltage(0.0f);
    qube.update();
  }
}

void setup() {
  Serial.begin(115200);
  qube.begin();
  enableMotor(false);
  visualBootPattern();
  qube.setRGB(0, 0, 999);
  qube.update();
  Serial.println("Place rotary arm at physical center and let pendulum hang down.");
  Serial.println("Encoder zeroing starts in 5 seconds...");
  delay(5000);

  qube.setRGB(0, 500, 0);
  qube.resetMotorEncoder();

  // Start with the pendulum hanging down when powering on/resetting.
  // This maps the bottom position to alpha=pi for the policy.
  qube.resetPendulumEncoder();
  qube.update();
  delay(500);

  previousTheta = readMotorAngleRad();
  previousAlpha = readPendulumAngleRadUprightZero();
  filteredThetaDot = 0.0f;
  filteredAlphaDot = 0.0f;
  previousMicros = micros();
  controlStartMillis = millis();

  Serial.println("QUBE policy runner ready.");
  Serial.print("RUN_MODE=");
  Serial.println(RUN_MODE);
  Serial.println("All control modes are designed to start from the hanging-down position.");
  Serial.println("Extra hardware policy voltage limit is +/-2.0 V.");
}

void stopForever(const char* reason, float theta, float alpha, float voltage) {
  writeMotorVoltage(0.0f);
  enableMotor(false);
  qube.setRGB(999, 0, 0);
  qube.update();

  Serial.print("STOP: ");
  Serial.println(reason);
  Serial.print("theta=");
  Serial.print(theta);
  Serial.print(" alpha=");
  Serial.print(alpha);
  Serial.print(" voltage=");
  Serial.print(voltage);
  Serial.print(" ampFault=");
  Serial.print(qube.hasAmplifierFault());
  Serial.print(" stall=");
  Serial.print(qube.hasStallDetected());
  Serial.print(" stallError=");
  Serial.println(qube.hasStallError());

  while (true) {
    Serial.print("STOP still active: ");
    Serial.println(reason);
    delay(100);
  }
}

bool enforceRecoverableArmLimit(float theta, float alpha, float thetaDot, float alphaDot, float activeArmLimit) {
  if (!softLimitActive && fabs(theta) <= activeArmLimit) {
    return false;
  }

  if (fabs(theta) > activeArmLimit) {
    softLimitActive = true;
  }

  if (softLimitActive && fabs(theta) < SOFT_LIMIT_RESUME_RAD) {
    softLimitActive = false;
    qube.setRGB(0, 500, 0);
    qube.update();
    Serial.println("RECOVER: arm returned inside soft limit, resuming controller.");
    return false;
  }

  writeMotorVoltage(0.0f);
  qube.setRGB(999, 500, 0);
  qube.update();

  static int printCounter = 0;
  if (++printCounter >= 20) {
    printCounter = 0;
    printTelemetry("SOFT_LIMIT_WAIT", theta, alpha, thetaDot, alphaDot, 0.0f);
  }
  return true;
}

void printTelemetry(const char* label, float theta, float alpha, float thetaDot, float alphaDot, float voltage) {
  Serial.print(label);
  Serial.print(" theta=");
  Serial.print(theta, 5);
  Serial.print(" alpha=");
  Serial.print(alpha, 5);
  Serial.print(" thetaDot=");
  Serial.print(thetaDot, 5);
  Serial.print(" alphaDot=");
  Serial.print(alphaDot, 5);
  Serial.print(" voltage=");
  Serial.print(voltage, 5);
  Serial.print(" motorDeg=");
  Serial.print(qube.getMotorAngle(false));
  Serial.print(" pendDeg=");
  Serial.print(qube.getPendulumAngle(false));
  Serial.print(" ampFault=");
  Serial.print(qube.hasAmplifierFault());
  Serial.print(" stall=");
  Serial.print(qube.hasStallDetected());
  Serial.print(" stallError=");
  Serial.println(qube.hasStallError());
}

void runSensorMode(float theta, float alpha, float thetaDot, float alphaDot) {
  writeMotorVoltage(0.0f);
  printTelemetry("SENSOR", theta, alpha, thetaDot, alphaDot, 0.0f);
  delay(100);
}

void runMotorSignMode(float theta, float alpha, float thetaDot, float alphaDot) {
  static bool positive = true;
  static unsigned long lastSwitch = 0;
  unsigned long now = millis();
  if (now - lastSwitch > 1000) {
    positive = !positive;
    lastSwitch = now;
  }
  float voltage = positive ? 0.5f : -0.5f;
  writeMotorVoltage(voltage);
  printTelemetry(positive ? "MOTOR_POS" : "MOTOR_NEG", theta, alpha, thetaDot, alphaDot, voltage);
  delay(100);
}

void runPolicyDryRunMode(float theta, float alpha, float thetaDot, float alphaDot) {
  float proposedVoltage = qubePolicyPredictVoltage(theta, alpha, thetaDot, alphaDot);
  writeMotorVoltage(0.0f);
  printTelemetry("POLICY_DRY_RUN", theta, alpha, thetaDot, alphaDot, proposedVoltage);
  delay(100);
}

void runPolicyEdgeDiagnosticMode(float theta, float alpha, float thetaDot, float alphaDot) {
  writeMotorVoltage(0.0f);

  float measuredVoltage = qubePolicyPredictVoltage(theta, alpha, thetaDot, alphaDot);
  float centerDownVoltage = qubePolicyPredictVoltage(0.0f, 3.14159265f, 0.0f, 0.0f);
  float leftEdgeDownVoltage = qubePolicyPredictVoltage(-1.2f, 3.14159265f, 0.0f, 0.0f);
  float rightEdgeDownVoltage = qubePolicyPredictVoltage(1.2f, 3.14159265f, 0.0f, 0.0f);

  printTelemetry("POLICY_EDGE_DIAG", theta, alpha, thetaDot, alphaDot, measuredVoltage);
  Serial.print("  synthetic_center_down_v=");
  Serial.print(centerDownVoltage, 5);
  Serial.print(" left_edge_down_v=");
  Serial.print(leftEdgeDownVoltage, 5);
  Serial.print(" right_edge_down_v=");
  Serial.println(rightEdgeDownVoltage, 5);
  delay(250);
}

void runOpenLoopSwingPowerTest(float theta, float alpha, float thetaDot, float alphaDot) {
  float timeSeconds = millis() * 0.001f;
  float centeredTheta = theta - CENTER_TRIM_RAD;
  float voltage = -CENTER_TEST_THETA_GAIN * centeredTheta - CENTER_TEST_THETA_DOT_GAIN * thetaDot;

  if (fabs(theta) < OPEN_LOOP_PUMP_REGION_RAD) {
    voltage += OPEN_LOOP_TEST_VOLTAGE * sin(2.0f * 3.14159265f * OPEN_LOOP_TEST_FREQUENCY_HZ * timeSeconds);
  } else if (fabs(theta) > OPEN_LOOP_CENTER_REGION_RAD) {
    voltage = -CENTER_TEST_THETA_GAIN * centeredTheta - CENTER_TEST_THETA_DOT_GAIN * thetaDot;
  }

  voltage = constrain(voltage, -OPEN_LOOP_TEST_VOLTAGE, OPEN_LOOP_TEST_VOLTAGE);
  writeMotorVoltage(voltage);

  static int printCounter = 0;
  if (++printCounter >= 20) {
    printCounter = 0;
    printTelemetry("OPEN_LOOP_POWER", theta, alpha, thetaDot, alphaDot, voltage);
  }
}

void runArmCenteringSignTest(float theta, float alpha, float thetaDot, float alphaDot) {
  float centeredTheta = theta - CENTER_TRIM_RAD;
  float voltage = 0.0f;
  if (fabs(centeredTheta) > CENTER_TEST_DEADBAND_RAD) {
    voltage = -CENTER_TEST_THETA_GAIN * centeredTheta - CENTER_TEST_THETA_DOT_GAIN * thetaDot;
  }
  voltage = constrain(voltage, -CENTER_TEST_VOLTAGE_LIMIT, CENTER_TEST_VOLTAGE_LIMIT);
  writeMotorVoltage(voltage);

  static int printCounter = 0;
  if (++printCounter >= 20) {
    printCounter = 0;
    printTelemetry("CENTER_SIGN_TEST", theta, alpha, thetaDot, alphaDot, voltage);
  }
}

float classicalBalanceVoltage(float theta, float alpha, float thetaDot, float alphaDot) {
  // Balance gains used after the swing-up reaches the top region.
  const float K_THETA = -1.93f;
  const float K_ALPHA = 33.40f;
  const float K_THETA_DOT = -1.40f;
  const float K_ALPHA_DOT = 3.08f;
  float voltage = -(K_THETA * theta + K_ALPHA * alpha + K_THETA_DOT * thetaDot + K_ALPHA_DOT * alphaDot);
  return constrain(voltage, -RUNNER_VOLTAGE_LIMIT, RUNNER_VOLTAGE_LIMIT);
}

float swingupVoltage(float theta, float alpha, float thetaDot, float alphaDot) {
  // Hold each push long enough to build pendulum energy instead of buzzing.
  unsigned long phase = (millis() / SWINGUP_PULSE_HALF_PERIOD_MS) % 2;
  float voltage = phase == 0 ? SWINGUP_VOLTAGE_LIMIT : -SWINGUP_VOLTAGE_LIMIT;
  voltage -= SWINGUP_CENTERING_THETA_GAIN * theta + SWINGUP_CENTERING_THETA_DOT_GAIN * thetaDot;
  return constrain(voltage, -SWINGUP_VOLTAGE_LIMIT, SWINGUP_VOLTAGE_LIMIT);
}

float hybridSwingupBalanceVoltage(float theta, float alpha, float thetaDot, float alphaDot) {
  if (fabs(alpha) < BALANCE_SWITCH_RAD) {
    return classicalBalanceVoltage(theta, alpha, thetaDot, alphaDot);
  }
  return swingupVoltage(theta, alpha, thetaDot, alphaDot);
}

float applyEdgeCenteringAssist(float voltage, float theta, float thetaDot) {
  float centeredTheta = theta - CENTER_TRIM_RAD;
  if (!ENABLE_EDGE_CENTERING_ASSIST || fabs(centeredTheta) < EDGE_ASSIST_START_RAD) {
    return voltage;
  }
  float edgeDepth = (fabs(centeredTheta) - EDGE_ASSIST_START_RAD) / (SWINGUP_ARM_LIMIT_RAD - EDGE_ASSIST_START_RAD);
  edgeDepth = constrain(edgeDepth, 0.0f, 1.0f);
  float centeringVoltage = -EDGE_ASSIST_THETA_GAIN * centeredTheta - EDGE_ASSIST_THETA_DOT_GAIN * thetaDot;
  return constrain((1.0f - edgeDepth) * voltage + edgeDepth * centeringVoltage, -RUNNER_VOLTAGE_LIMIT, RUNNER_VOLTAGE_LIMIT);
}

float applyMinimumPolicyKick(float voltage) {
  if (fabs(voltage) < 0.001f) {
    return 0.0f;
  }
  if (fabs(voltage) >= RL_POLICY_MIN_ABS_VOLTAGE) {
    return voltage;
  }
  return voltage > 0.0f ? RL_POLICY_MIN_ABS_VOLTAGE : -RL_POLICY_MIN_ABS_VOLTAGE;
}

float assistedRlVoltage(float theta, float alpha, float thetaDot, float alphaDot) {
  if (USE_CLASSICAL_BALANCE_IN_MODE3) {
    if (!balanceModeActive && fabs(alpha) < BALANCE_ENTER_ALPHA_RAD) {
      balanceModeActive = true;
    } else if (balanceModeActive && fabs(alpha) > BALANCE_EXIT_ALPHA_RAD) {
      balanceModeActive = false;
    }

    if (balanceModeActive) {
      return classicalBalanceVoltage(theta, alpha, thetaDot, alphaDot);
    }
  }

  if (USE_SWINGUP_ASSIST_IN_MODE3 && fabs(alpha) > RL_HANDOVER_ALPHA_RAD) {
    return swingupVoltage(theta, alpha, thetaDot, alphaDot);
  }

  float policyTheta = theta - CENTER_TRIM_RAD;
  float voltage = RL_POLICY_FORCE_GAIN * qubePolicyPredictVoltage(policyTheta, alpha, thetaDot, alphaDot);
  voltage = applyMinimumPolicyKick(voltage);
  return voltage;
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
  float rawThetaDot = (theta - previousTheta) / dt;
  float rawAlphaDot = wrapRad(alpha - previousAlpha) / dt;
  filteredThetaDot = (1.0f - VELOCITY_FILTER_ALPHA) * filteredThetaDot + VELOCITY_FILTER_ALPHA * rawThetaDot;
  filteredAlphaDot = (1.0f - VELOCITY_FILTER_ALPHA) * filteredAlphaDot + VELOCITY_FILTER_ALPHA * rawAlphaDot;
  float thetaDot = filteredThetaDot;
  float alphaDot = filteredAlphaDot;
  previousTheta = theta;
  previousAlpha = alpha;

  if (RUN_MODE == 0) {
    runSensorMode(theta, alpha, thetaDot, alphaDot);
    return;
  }

  if (RUN_MODE == 1) {
    runMotorSignMode(theta, alpha, thetaDot, alphaDot);
    return;
  }

  if (RUN_MODE == 4) {
    runPolicyDryRunMode(theta, alpha, thetaDot, alphaDot);
    return;
  }

  if (RUN_MODE == 5) {
    runPolicyEdgeDiagnosticMode(theta, alpha, thetaDot, alphaDot);
    return;
  }

  float activeArmLimit = RUN_MODE == 2 || RUN_MODE == 3 || RUN_MODE == 4 || RUN_MODE == 5 || RUN_MODE == 6 || RUN_MODE == 7 ? SWINGUP_ARM_LIMIT_RAD : ARM_LIMIT_RAD;
  if (enforceRecoverableArmLimit(theta, alpha, thetaDot, alphaDot, activeArmLimit)) {
    return;
  }

  float voltage = 0.0f;
  if (RUN_MODE == 7) {
    runArmCenteringSignTest(theta, alpha, thetaDot, alphaDot);
    return;
  } else if (RUN_MODE == 6) {
    runOpenLoopSwingPowerTest(theta, alpha, thetaDot, alphaDot);
    return;
  } else if (RUN_MODE == 2) {
    voltage = hybridSwingupBalanceVoltage(theta, alpha, thetaDot, alphaDot);
  } else {
    float elapsedSeconds = (millis() - controlStartMillis) * 0.001f;
    if (elapsedSeconds < MODE3_STARTUP_TEST_SECONDS) {
      voltage = MODE3_STARTUP_TEST_VOLTAGE;
    } else {
      voltage = assistedRlVoltage(theta, alpha, thetaDot, alphaDot);
      voltage = applyEdgeCenteringAssist(voltage, theta, thetaDot);
      voltage = constrain(voltage, -RUNNER_VOLTAGE_LIMIT, RUNNER_VOLTAGE_LIMIT);
    }
  }
  if ((RUN_MODE == 2 && fabs(alpha) < BALANCE_SWITCH_RAD) || (RUN_MODE == 3 && balanceModeActive)) {
    showBalanceMode(voltage);
  } else {
    showVoltageDirection(voltage);
  }
  writeMotorVoltage(voltage);

  if (qube.hasAmplifierFault() || qube.hasStallError()) {
    stopForever("QUBE amplifier fault or stall error", theta, alpha, voltage);
  }

  static int printCounter = 0;
  if (++printCounter >= 20) {
    printCounter = 0;
    if (RUN_MODE == 2) {
      printTelemetry(fabs(alpha) < BALANCE_SWITCH_RAD ? "HYBRID_BALANCE" : "HYBRID_SWINGUP", theta, alpha, thetaDot, alphaDot, voltage);
    } else {
      printTelemetry("POLICY", theta, alpha, thetaDot, alphaDot, voltage);
    }
  }
}
