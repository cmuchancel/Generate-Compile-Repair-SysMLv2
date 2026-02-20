package SensorManagementSystem {
    private import ScalarValues::*;

    enum def SensorStatus {
        enum READY;
        enum NOT_READY;
        enum UPDATING;
        enum UPDATED;
    }

    item def SensorData {
        attribute dataValue : Real;
        attribute timestamp : Real;
    }

    port def SensorPort {
        attribute portId : Integer;
    }

    port def UpdatePort {
        attribute updateTriggered : Boolean;
    }

    part def Sensor {
        attribute sensorId : Integer;
        attribute sensorName : String;
        attribute status : SensorStatus;
        attribute isReady : Boolean;
        attribute lastUpdateTime : Real;
        attribute updateCount : Integer;

        port sensorOut : SensorPort;
        port updateIn : UpdatePort;
    }

    part def SensorRegistry {
        attribute registryId : Integer;
        attribute totalSensors : Integer;
        attribute activeSensors : Integer;

        part sensor1 : Sensor {
            attribute sensorId : Integer = 1;
            attribute sensorName : String = "Sensor1";
            attribute status : SensorStatus = SensorStatus::READY;
            attribute isReady : Boolean = true;
            attribute lastUpdateTime : Real = 0.0;
            attribute updateCount : Integer = 0;

            port sensorOut : SensorPort;
            port updateIn : UpdatePort;
        }

        part sensor2 : Sensor {
            attribute sensorId : Integer = 2;
            attribute sensorName : String = "Sensor2";
            attribute status : SensorStatus = SensorStatus::NOT_READY;
            attribute isReady : Boolean = false;
            attribute lastUpdateTime : Real = 0.0;
            attribute updateCount : Integer = 0;

            port sensorOut : SensorPort;
            port updateIn : UpdatePort;
        }

        part sensor3 : Sensor {
            attribute sensorId : Integer = 3;
            attribute sensorName : String = "Sensor3";
            attribute status : SensorStatus = SensorStatus::READY;
            attribute isReady : Boolean = true;
            attribute lastUpdateTime : Real = 0.0;
            attribute updateCount : Integer = 0;

            port sensorOut : SensorPort;
            port updateIn : UpdatePort;
        }
    }

    part def UpdateController {
        attribute controllerId : Integer;
        attribute controllerName : String;
        attribute monitoringActive : Boolean;
        attribute traversalCount : Integer;
        attribute updateThreshold : Real;

        port controllerUpdateOut : UpdatePort;
        port controllerSensorIn : SensorPort;
    }

    part def StatusMonitor {
        attribute monitorId : Integer;
        attribute monitorName : String;
        attribute pollInterval : Real;
        attribute readySensorsCount : Integer;
        attribute processedSensorsCount : Integer;

        port monitorSensorIn : SensorPort;
        port monitorUpdateOut : UpdatePort;
    }

    part def SensorManagementSystemRoot {
        attribute systemId : Integer = 1;
        attribute systemName : String = "SensorManagementSystem";
        attribute systemVersion : Real = 1.0;
        attribute autoUpdateEnabled : Boolean = true;
        attribute reliabilityScore : Real = 0.99;
        attribute responsivenessLevel : Integer = 1;

        part registry : SensorRegistry {
            attribute registryId : Integer = 1;
            attribute totalSensors : Integer = 3;
            attribute activeSensors : Integer = 2;

            part sensor1 : Sensor {
                attribute sensorId : Integer = 1;
                attribute sensorName : String = "Sensor1";
                attribute status : SensorStatus = SensorStatus::READY;
                attribute isReady : Boolean = true;
                attribute lastUpdateTime : Real = 0.0;
                attribute updateCount : Integer = 0;

                port sensorOut : SensorPort;
                port updateIn : UpdatePort;
            }

            part sensor2 : Sensor {
                attribute sensorId : Integer = 2;
                attribute sensorName : String = "Sensor2";
                attribute status : SensorStatus = SensorStatus::NOT_READY;
                attribute isReady : Boolean = false;
                attribute lastUpdateTime : Real = 0.0;
                attribute updateCount : Integer = 0;

                port sensorOut : SensorPort;
                port updateIn : UpdatePort;
            }

            part sensor3 : Sensor {
                attribute sensorId : Integer = 3;
                attribute sensorName : String = "Sensor3";
                attribute status : SensorStatus = SensorStatus::READY;
                attribute isReady : Boolean = true;
                attribute lastUpdateTime : Real = 0.0;
                attribute updateCount : Integer = 0;

                port sensorOut : SensorPort;
                port updateIn : UpdatePort;
            }
        }

        part controller : UpdateController {
            attribute controllerId : Integer = 1;
            attribute controllerName : String = "MainUpdateController";
            attribute monitoringActive : Boolean = true;
            attribute traversalCount : Integer = 0;
            attribute updateThreshold : Real = 1.0;

            port controllerUpdateOut : UpdatePort;
            port controllerSensorIn : SensorPort;
        }

        part monitor : StatusMonitor {
            attribute monitorId : Integer = 1;
            attribute monitorName : String = "MainStatusMonitor";
            attribute pollInterval : Real = 0.5;
            attribute readySensorsCount : Integer = 0;
            attribute processedSensorsCount : Integer = 0;

            port monitorSensorIn : SensorPort;
            port monitorUpdateOut : UpdatePort;
        }

        connect registry.sensor1.sensorOut to monitor.monitorSensorIn;
        connect registry.sensor2.sensorOut to monitor.monitorSensorIn;
        connect registry.sensor3.sensorOut to monitor.monitorSensorIn;
        connect monitor.monitorUpdateOut to controller.controllerSensorIn;
        connect controller.controllerUpdateOut to registry.sensor1.updateIn;
        connect controller.controllerUpdateOut to registry.sensor2.updateIn;
        connect controller.controllerUpdateOut to registry.sensor3.updateIn;
    }
}