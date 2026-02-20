package SensorSystem {
    private import ScalarValues::*;

    enum def SensorStatus {
        Ready,
        Updating,
        Updated,
        Error
    };

    part def Sensor {
        attribute id : String;
        attribute status : SensorStatus;
        attribute lastUpdateTime : Integer = 0;
    }

    part def SensorManager {
        part sensors : Sensor[0..*];
        attribute updateInterval : Integer = 1000;
    }

    part system : SensorManager;
}