# QUBE Down-Start Calibration Checklist

The project assumes the QUBE starts with the pendulum hanging down. The goal is:

- start down
- swing up
- balance
- swing up again if it falls

## 1. Sensor Mode

Set:

```cpp
const int RUN_MODE = 0;
```

Upload and open Serial Monitor at `115200`.

Expected after boot with the pendulum hanging down:

- `alpha` should be close to `pi` or `-pi`.
- Moving the pendulum toward the top should move `alpha` toward `0`.
- Moving the arm should change `theta` smoothly.

If signs look opposite to the simulator, flip one constant at a time:

```cpp
const float THETA_SIGN = -1.0f;
const float ALPHA_SIGN = -1.0f;
```

## 2. Motor Sign Mode

Set:

```cpp
const int RUN_MODE = 1;
```

This sends only `+/-0.5 V`. It will not swing up the pendulum.

If positive voltage produces the wrong positive/negative `theta` direction, flip:

```cpp
const float MOTOR_VOLTAGE_SIGN = -1.0f;
```

## 3. Classical Hybrid Swing-Up

Set:

```cpp
const int RUN_MODE = 2;
```

This uses a simple energy-pumping swing-up heuristic far from the top and switches to balance near the top. This is a diagnostic controller, not the trained PPO policy.

## 4. RL Swing-Up Policy

Only use this after exporting a policy trained from the down position:

```cpp
const int RUN_MODE = 3;
```

Start with conservative voltage limits and increase only after signs and sensor offsets are correct.

## 5. Policy Dry-Run / Edge Check

Set:

```cpp
const int RUN_MODE = 5;
```

This does not power the motor. It prints the voltage the exported policy would
apply for the measured state, plus synthetic reference values.

For the current `sac_qube_swingup_90` policy at `+/-2 V`, the synthetic values
should be approximately:

- center/down: `+0.59 V`
- left edge/down (`theta=-1.2 rad`): `+1.58 V`
- right edge/down (`theta=+1.2 rad`): `-1.51 V`

If the measured hardware near the physical left/right edge disagrees with the
expected sign, fix `THETA_SIGN`. If the sign looks right in dry-run but powered
mode drives farther into the edge, fix `MOTOR_VOLTAGE_SIGN`.

Do not increase to `5 V` until this dry-run pattern and the low-voltage motor
sign test both agree.

## 6. Open-Loop Power Test

If Serial Monitor is unavailable, use this mode to test whether the motor can
physically inject enough energy without depending on the RL policy:

```cpp
const int RUN_MODE = 6;
```

This applies a timed rocking command with light arm-centering. Keep a hand
near the emergency stop. If this mode also cannot make the pendulum swing
substantially, the problem is not the RL policy; check motor voltage scaling,
motor sign, power supply/current limits, and mechanical friction.

Only after this produces a strong swing should you try increasing the RL policy
voltage or retraining at a higher simulated voltage limit.

## 7. Arm Centering Sign Test

If Serial Monitor does not work, use this before any swing-up test:

```cpp
const int RUN_MODE = 7;
```

Hold the pendulum safely so it cannot swing up during this test. Manually move
the rotary arm about 20-30 degrees left or right of center before releasing the
arm. The arm should slowly move back toward the physical center.

If it moves farther toward the nearest stop, flip exactly one of these and try
again:

```cpp
const float MOTOR_VOLTAGE_SIGN = -1.0f;
```

If that does not fix it, restore `MOTOR_VOLTAGE_SIGN` and flip:

```cpp
const float THETA_SIGN = -1.0f;
```

Do not continue with `RUN_MODE = 3` until this center test moves toward center
from both sides.

If it centers correctly but consistently settles slightly off to the same side,
trim the desired center in small steps:

```cpp
const float CENTER_TRIM_DEG = 3.0f;
```

Use `2-3 degree` steps. Positive trim moves the desired center toward positive
`theta`; negative trim moves it toward negative `theta`.

If the center is correct but the motor seems to have a one-sided deadband or
friction bias, use a very small bias:

```cpp
const float MOTOR_BIAS_VOLTAGE = 0.05f;
```

Keep this small. Values above about `0.15 V` usually mean something else is
wrong, such as sign, zeroing, friction, or wiring/current limits.
