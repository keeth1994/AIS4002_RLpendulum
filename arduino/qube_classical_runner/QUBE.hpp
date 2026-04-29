#ifndef QUBE_HPP
#define QUBE_HPP
#define CW 1
#define CCW 0

#include <Arduino.h>
#include <SPI.h>

class QUBE
{
public:
    QUBE(int cs = 10);

    byte output[17];
    uint32_t input[6];
    int CS;

    void print();
    void setRGB(int r, int g, int b);
    void setMotorSpeed(int v);
    void setMotorVoltage(float V);
    void resetMotorEncoder();
    void resetPendulumEncoder();
    void setMotorEncoder(int count);
    void setPendulumEncoder(int count);
    void begin();
    void update();
    float getMotorCurrent();
    float getRPM();
    long getMotorEncoder();
    long getPendulumEncoder();
    float getMotorAngle(bool absolute = true);
    float getPendulumAngle(bool absolute = true);
    float getMotorVoltage();
    bool hasAmplifierFault();
    bool hasStallDetected();
    bool hasStallError();

private:
    int lastColor[3];
    bool amplifierFault;
    bool stallDetected;
    bool stallError;
    long LEDBlinkTimer = 0;
    bool LED_ON;

    float lastMotorCount = 0;
    bool spinDir;

    float voltage;

    void checkStatus();
    void setErrorLight();
};

#endif
