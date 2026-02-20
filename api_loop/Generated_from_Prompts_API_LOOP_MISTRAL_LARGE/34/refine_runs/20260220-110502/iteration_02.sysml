package SensorManagement {
    private import ScalarValues::*;

    enum def SensorState {
        enum READY;
        enum UPDATING;
        enum ERROR;
        enum IDLE;
    }

    part def Sensor {
        attribute id : String;
        attribute sensorState : SensorState = SensorState::IDLE;
        attribute lastUpdate : Real;
    }

    part def SensorManager {
        attribute sensors : Sensor[0..*];
        attribute updateInterval : Real = 1.0;

        port updatePort : UpdatePort;
    }

    port def UpdatePort;

    item def UpdateCommand {
        attribute targetSensor : Sensor;
    }

    part system : SensorManager;

    part updateCommand : UpdateCommand {
        attribute targetSensor : Sensor;
    }

    connect updateCommand to system.updatePort;
}