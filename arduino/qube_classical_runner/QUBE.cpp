#include "QUBE.hpp"

void QUBE::print()
{
    Serial.print("Motor: ");
    Serial.print(getMotorAngle());
    Serial.print("    Pendulum: ");
    Serial.print(getPendulumAngle());
    Serial.print("    RPM: ");
    Serial.print(getRPM());
    Serial.print("    Current (mA): ");
    Serial.print(getMotorCurrent());
    Serial.print("    Motor voltage (V):");
    Serial.print(getMotorVoltage());
    Serial.print("    Watts (W):");
    Serial.print(getMotorVoltage() * getMotorCurrent() * 0.001);
    Serial.print("    Amplifier error: ");
    Serial.print(amplifierFault);
    Serial.print("    Stall detected: ");
    Serial.print(stallDetected);
    Serial.print("    Stall Error: ");
    Serial.print(stallError);
    Serial.println();
}

QUBE::QUBE(int cs)
{
    CS = cs;
    pinMode(CS, OUTPUT);
    digitalWrite(CS, HIGH);

    for (int i = 0; i < 17; ++i)
    {
        output[i] = 0;
    }
    for (int i = 0; i < 6; ++i)
    {
        input[i] = 0;
    }
    lastColor[0] = 0;
    lastColor[1] = 0;
    lastColor[2] = 0;
    amplifierFault = false;
    stallDetected = false;
    stallError = false;
    LED_ON = false;
    spinDir = true;
    voltage = 0.0;

    output[0] = 1;
    output[1] = 0;
    output[2] = 0;
}

void QUBE::setRGB(int r, int g, int b)
{
    r = constrain(r, 0, 999);
    byte R_MSB = r >> 8;
    byte R_LSB = r;

    g = constrain(g, 0, 999);
    byte G_MSB = g >> 8;
    byte G_LSB = g;

    b = constrain(b, 0, 999);
    byte B_MSB = b >> 8;
    byte B_LSB = b;

    lastColor[0] = r;
    lastColor[1] = g;
    lastColor[2] = b;

    output[2] |= B00011100;
    output[3] = R_MSB;
    output[4] = R_LSB;
    output[5] = G_MSB;
    output[6] = G_LSB;
    output[7] = B_MSB;
    output[8] = B_LSB;
}

void QUBE::setErrorLight()
{
    long now = micros();
    if (now - LEDBlinkTimer > 1e6 * 0.25)
    {
        LEDBlinkTimer += 1e6 * 0.25;
        LED_ON = !LED_ON;
    }

    int r = 0;
    int g = 0;
    int b = 0;

    if (stallDetected)
    {
        r = 999;
    }

    if (stallError)
    {
        r = 999 * LED_ON;
    }

    if (amplifierFault)
    {
        r = (now % 1000000) * 1e-3;
        r = constrain(r, 0, 999);
    }

    output[2] |= B00011100;
    output[3] = r >> 8;
    output[4] = r;
    output[5] = g >> 8;
    output[6] = g;
    output[7] = b >> 8;
    output[8] = b;
}

void QUBE::setMotorSpeed(int v)
{
    bool dir = v >= 0;

    v = constrain(v, -999, 999);
    voltage = 24.0 * v / 999.0;

    v += (1 << (16 - dir));

    byte v_MSB = v >> 8;
    byte v_LSB = v;

    output[2] |= B00000011;
    output[15] = v_MSB;
    output[16] = v_LSB;
}

void QUBE::setMotorVoltage(float V)
{
    V = constrain(V, -24.0, 24.0);
    int pwm_duty_cycle_10x = (V / 24.0) * 999.0;
    setMotorSpeed(pwm_duty_cycle_10x);
}

void QUBE::resetMotorEncoder()
{
    setMotorEncoder(0);
}

void QUBE::resetPendulumEncoder()
{
    setPendulumEncoder(0);
}

void QUBE::setMotorEncoder(int count)
{
    output[2] |= B00100000;
    uint32_t value = ((uint32_t)count) & 0x00FFFFFF;
    output[9] = (value >> 16) & 0xFF;
    output[10] = (value >> 8) & 0xFF;
    output[11] = value & 0xFF;
}

void QUBE::setPendulumEncoder(int count)
{
    output[2] |= B01000000;
    uint32_t value = ((uint32_t)count) & 0x00FFFFFF;
    output[12] = (value >> 16) & 0xFF;
    output[13] = (value >> 8) & 0xFF;
    output[14] = value & 0xFF;
}

void QUBE::begin()
{
    SPI.begin();
}

long QUBE::getMotorEncoder()
{
    long data = input[1];
    bool negative = input[1] >> 23;
    if (negative)
        data -= 0x1000000;
    return data;
}

long QUBE::getPendulumEncoder()
{
    long data = input[2];
    bool negative = input[2] >> 23;
    if (negative)
        data -= 0x1000000;
    return data;
}

float QUBE::getMotorAngle(bool absolute)
{
    long count = getMotorEncoder();

    if (absolute)
    {
        count %= 2048;
        if (count <= -1024)
        {
            count += 2048;
        }

        if (count > 1024)
        {
            count -= 2048;
        }
    }

    float angle = ((float)count / 2048.0) * 360.0;
    return angle;
}

float QUBE::getPendulumAngle(bool absolute)
{
    long count = getPendulumEncoder();
    if (absolute)
    {
        count %= 2048;
        if (count <= -1024)
        {
            count += 2048;
        }

        if (count > 1024)
        {
            count -= 2048;
        }
    }

    float angle = ((float)count / 2048.0) * 360.0;
    return angle;
}

float QUBE::getRPM()
{
    uint32_t tach = input[3];

    if (tach == 0 || tach == 0x7FFFFF || tach == 0xFFFFFF)
    {
        return 0;
    }

    bool tachDir = (tach >> 23);
    tach = tach & 0x7FFFFF;
    float rpm = (4.0 / (tach * 25.0 * pow(10, -9))) * 60 / 512 / 4;
    tachDir ? rpm = rpm : rpm = -rpm;
    return rpm;
}

float QUBE::getMotorVoltage()
{
    return voltage;
}

bool QUBE::hasAmplifierFault()
{
    return amplifierFault;
}

bool QUBE::hasStallDetected()
{
    return stallDetected;
}

bool QUBE::hasStallError()
{
    return stallError;
}

void QUBE::checkStatus()
{
    byte status = input[4];

    amplifierFault = status & B00000001;
    stallDetected = status & B00000010;
    stallError = status & B00000100;
}

float QUBE::getMotorCurrent()
{
    float current = (input[5] - 8190.0) / 9828.0;
    return current * 1000;
}

void QUBE::update()
{
    checkStatus();
    if (stallError || amplifierFault)
    {
        setErrorLight();
    }
    else
    {
        setRGB(lastColor[0], lastColor[1], lastColor[2]);
    }

    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE2));
    digitalWrite(CS, LOW);

    int ID = SPI.transfer(output[0]) << 8;
    ID |= SPI.transfer(output[1]);

    byte encoder01 = SPI.transfer(output[2]);
    byte encoder02 = SPI.transfer(output[3]);
    byte encoder03 = SPI.transfer(output[4]);

    byte encoder11 = SPI.transfer(output[5]);
    byte encoder12 = SPI.transfer(output[6]);
    byte encoder13 = SPI.transfer(output[7]);

    byte tach1 = SPI.transfer(output[8]);
    byte tach2 = SPI.transfer(output[9]);
    byte tach3 = SPI.transfer(output[10]);
    byte status = SPI.transfer(output[11]);

    int currentSense = SPI.transfer(output[12]) << 8;
    currentSense |= SPI.transfer(output[13]);

    SPI.transfer(output[14]);
    SPI.transfer(output[15]);
    SPI.transfer(output[16]);

    digitalWrite(CS, HIGH);
    SPI.endTransaction();

    input[0] = ID;
    input[1] = ((uint32_t)encoder01) << 16 | ((uint32_t)encoder02) << 8 | encoder03;
    input[2] = ((uint32_t)encoder11) << 16 | ((uint32_t)encoder12) << 8 | encoder13;
    input[3] = ((uint32_t)tach1) << 16 | ((uint32_t)tach2) << 8 | tach3;
    input[4] = status;
    input[5] = currentSense;

    output[2] = 0;
}
