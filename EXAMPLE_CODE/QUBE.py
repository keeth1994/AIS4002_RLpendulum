import serial
import time

RESET_ENCODER_0 = 64
RESET_ENCODER_1 = 32
SET_LED_RED = 16
SET_LED_GREEN = 8
SET_LED_BLUE = 4
SET_MOTOR_SPEED = 3


def constrain(val, _min, _max):
    return min(max(val, _min), _max)


class QUBE:
    def __init__(self, port, baudrate):
        self.master = serial.Serial(
            port=port, baudrate=baudrate, timeout=0.1, bytesize=serial.EIGHTBITS
        )
        self.rpm = 0
        self.voltage = 0
        self.current = 0
        self.motorAngle = 0
        self.pendulumAngle = 0
        self.r = 0
        self.g = 0
        self.b = 0
        self.writemask = [0, 0, 0, 0, 0, 0]
        self.output = self._neutral_output_packet()
        self.input = []
        self.startTime = time.time()

    def _neutral_output_packet(self):
        # The Python_Serial.ino bridge interprets motor bytes as an always-active
        # command with an offset of +999. That means [0, 0] is full negative
        # drive, not "do nothing". Idle packets must explicitly encode zero speed.
        return [
            0,  # reset enc0 - 0
            0,  # reset enc1 - 1
            0,  # red msb - 2
            0,  # red lsb - 3
            0,  # green msb - 4
            0,  # green lsb - 5
            0,  # blue msb - 6
            0,  # blue lsb - 7
            3,  # motor msb for 0 command (999)
            231,  # motor lsb for 0 command (999)
        ]

    # Getters
    def getMotorAngle(self):
        return self.motorAngle

    def getPendulumAngle(self):
        return self.pendulumAngle

    def getMotorRPM(self):
        return self.rpm

    def getMotorCurrent(self):
        return self.current

    # Setters
    def resetMotorEncoder(self):
        self.output[0] = 0x01
        self.writemask[0] = RESET_ENCODER_0

    def resetPendulumEncoder(self):
        self.output[1] = 0x01
        self.writemask[1] = RESET_ENCODER_1

    def setMotorSpeed(self, speed):
        speed = constrain(speed, -999, 999)
        speed += 999

        speed = int(speed)
        speed_MSB = speed >> 8
        speed_LSB = speed & 0xFF
        self.output[8] = speed_MSB
        self.output[9] = speed_LSB
        self.writemask[5] = SET_MOTOR_SPEED

    def setMotorVoltage(self, volts):
        self.voltage = min(24, max(-24, volts))
        speed = (volts / 24.0) * 999
        self.setMotorSpeed(speed)

    def setRGB(self, r, g, b):
        self.r = constrain(r, 0, 999)
        r_MSB = self.r >> 8
        r_LSB = self.r & 0xFF
        self.output[2] = r_MSB
        self.output[3] = r_LSB

        self.g = constrain(g, 0, 999)
        g_MSB = self.g >> 8
        g_LSB = self.g & 0xFF
        self.output[4] = g_MSB
        self.output[5] = g_LSB

        self.b = constrain(b, 0, 999)
        b_MSB = self.b >> 8
        b_LSB = self.b & 0xFF
        self.output[6] = b_MSB
        self.output[7] = b_LSB

        self.writemask[2] = SET_LED_RED
        self.writemask[3] = SET_LED_GREEN
        self.writemask[4] = SET_LED_BLUE

    def reset_buffers(self):
        self.master.reset_input_buffer()
        self.master.reset_output_buffer()

    def _read_exactly(self, size):
        data = self.master.read(size)
        if len(data) != size:
            raise TimeoutError(
                f"Expected {size} reply bytes from QUBE bridge, received {len(data)}"
            )
        return data

    # Serial communication and bit manipulation
    def receiveEncoderAngle(self, packet, offset):
        rev_MSB = packet[offset + 0]
        rev_LSB = packet[offset + 1]
        ang_MSB = packet[offset + 2]
        ang_LSB = packet[offset + 3]

        dir = rev_MSB >> 7
        revolutions = (rev_LSB) + (rev_MSB << 8) - (dir * 2**15)

        angleInt = ang_MSB * 2 + (ang_LSB >> 7)
        angleDec = (ang_LSB & 0b01111111) * 0.01
        angle = angleInt + angleDec

        if dir:
            revolutions = -revolutions
            angle = -angle

        return revolutions * 360.0 + angle

    def receiveMotorRPM(self, packet, offset):
        rpm_MSB = packet[offset + 0]
        rpm_LSB = packet[offset + 1]
        dir = rpm_MSB >> 7
        rpm = ((rpm_MSB - (dir << 7)) << 8) | rpm_LSB
        if dir:
            rpm = -rpm
        return rpm

    def receiveMotorCurrent(self, packet, offset):
        current_MSB = packet[offset + 0]
        current_LSB = packet[offset + 1]

        current = (current_MSB << 8) | current_LSB
        return current

    def getLogData(self, motorSetpoint, pendulumSetpoint, rpm_target):
        return [
            self.motorAngle,
            motorSetpoint,
            self.pendulumAngle,
            pendulumSetpoint,
            self.rpm,
            rpm_target,
            self.voltage,
            self.current,
        ]

    def getPlotData(self, motorSetpoint, pendulumSetpoint, rpm_target):
        return [
            self.motorAngle,  # Plot 1 pink
            motorSetpoint,  # Plot 1 white
            self.pendulumAngle,  # Plot 2 pink
            pendulumSetpoint,  # Plot 2 white
            self.rpm,  # Plot 3 pink
            rpm_target,  # Plot 3 white
            self.voltage,  # Plot 4 pink
            self.current,
            time.time() - self.startTime,
        ]

    # Main update loop
    def update(self):
        data = []
        for byte in self.output:
            data.append(byte)
        self.master.write(bytearray(data))
        self.output = self._neutral_output_packet()
        packet = self._read_exactly(12)

        self.motorAngle = self.receiveEncoderAngle(packet, 0)
        self.pendulumAngle = self.receiveEncoderAngle(packet, 4)
        self.rpm = self.receiveMotorRPM(packet, 8)
        self.current = self.receiveMotorCurrent(packet, 10)
