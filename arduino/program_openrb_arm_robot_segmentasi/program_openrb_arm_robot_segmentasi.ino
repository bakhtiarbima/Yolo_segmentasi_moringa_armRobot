#include <Dynamixel2Arduino.h>

#define DXL_SERIAL Serial1  
#define DXL_DIR_PIN 2  
#define DXL_BAUDRATE 57600  

#define DXL_ID_1 1  // Base
#define DXL_ID_2 2  // Shoulder
#define DXL_ID_3 3  // Elbow
#define DXL_ID_4 4  // Wrist 1

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

// Dimensi Robot Arm (cm)
const float L1 = 13.5;
const float L2 = 13.5;
const float L4 = 10.0;
const float SHOULDER_HEIGHT = 10.0;

// Kalibrasi Offset
const float OFFSET_1 = +138;
const float OFFSET_2 = 0;
const float OFFSET_3 = 140.51;
const float OFFSET_4 = +70;

void inverseKinematics(float x, float y, float z, float* angles) {
    z -= SHOULDER_HEIGHT;
    angles[0] = atan2(y, x) * 180 / PI;

    float L = sqrt(x*x + y*y);
    float D = sqrt(L*L + (z - L4)*(z - L4));

    float cosTheta3 = (D*D - L1*L1 - L2*L2) / (2 * L1 * L2);
    angles[2] = -acos(constrain(cosTheta3, -1, 1)) * 180 / PI;

    float theta2a = atan2(z - L4, L);
    float theta2b = atan2(L2 * sin(-angles[2] * PI / 180), L1 + L2 * cos(-angles[2] * PI / 180));
    angles[1] = (theta2a + theta2b) * 180 / PI;

    angles[3] = -(angles[1] + angles[2]);

    angles[0] += OFFSET_1;
    angles[1] += OFFSET_2;
    angles[2] += OFFSET_3;
    angles[3] += OFFSET_4;
}

void moveServoWithSpeed(int id, float targetAngle, float speed) {
    int currentPos = dxl.getPresentPosition(id, UNIT_DEGREE);
    int goalPos = targetAngle;
    if (currentPos == goalPos) return;

    int stepSize = (goalPos > currentPos) ? 1 : -1;

    while (abs(currentPos - goalPos) > 5) {
        currentPos += stepSize;
        dxl.setGoalPosition(id, currentPos, UNIT_DEGREE);
        delay(speed);
    }
}

void executeMovement(int x_target, int y_target, float z_target, float speed) {
    float angles[4];
    inverseKinematics(x_target, y_target, z_target, angles);

    moveServoWithSpeed(DXL_ID_2, angles[1], speed);
    moveServoWithSpeed(DXL_ID_4, angles[3], speed);
    moveServoWithSpeed(DXL_ID_3, angles[2], speed);
    moveServoWithSpeed(DXL_ID_1, angles[0], speed);
}

void setup() {
    Serial.begin(115200);
    pinMode(A1, OUTPUT);

    dxl.begin(DXL_BAUDRATE);
    dxl.setPortProtocolVersion(2.0);

    for (int id = 1; id <= 4; id++) {
        dxl.torqueOff(id);
        dxl.setOperatingMode(id, OP_POSITION);
        dxl.torqueOn(id);
    }

    digitalWrite(A1, HIGH);
    Serial.println("Dynamixel Initialized!");
}

void loop() {
    if (Serial.available() > 0) {
        String data = Serial.readStringUntil('\n');
        int x_target, y_target;
        
        // Pastikan menerima data sebagai integer
        sscanf(data.c_str(), "%d,%d", &x_target, &y_target);

        Serial.print("Menerima koordinat: ");
        Serial.print("x: ");
        Serial.print(x_target);
        Serial.print("/n");
        Serial.print("y: ");
        Serial.println(y_target);

        executeMovement(0, -10, 31, 5); // posisi aman
        delay(2000);
        executeMovement(20, 0, 31, 5); // posisi awal1
        delay(1000);
        executeMovement(x_target, y_target, 31, 5);
        executeMovement(x_target, y_target, 22, 5); // posisi target
        Serial.println("sudah di posisi");
        digitalWrite(A1, LOW);
        delay(3000);
        executeMovement(-17, -10, 31, 5); // posisi akhir1
        executeMovement(-17, -10, 22, 5); // posisi pembuangan
        delay(200);
        digitalWrite(A1, HIGH);
        delay(3000);
        executeMovement(0, -10, 31, 5); // posisi aman
        delay(2000);

        Serial.println("arduino selesai");
    }
}
