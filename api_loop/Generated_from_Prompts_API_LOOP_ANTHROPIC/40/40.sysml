package VehicleHealthMonitoring {
    private import ScalarValues::*;

    enum def VehicleHealthStatus {
        enum NORMAL;
        enum DEGRADED;
        enum MAINTENANCE;
    }

    enum def WarningType {
        enum OVER_TEMPERATURE;
        enum NONE;
    }

    item def TemperatureReading {
        attribute value : Real;
        attribute unit : String;
    }

    item def MaintenanceRecord {
        attribute lastMaintenanceTime : Real;
        attribute nextMaintenanceTime : Real;
        attribute maintenanceDuration : Real;
    }

    item def WarningMessage {
        attribute warningType : WarningType;
        attribute timestamp : Real;
        attribute description : String;
    }

    port def TemperatureSensorPort {
        attribute currentTemperature : Real;
    }

    port def ControllerPort {
        attribute warningReceived : WarningType;
        attribute statusUpdate : VehicleHealthStatus;
    }

    port def MaintenancePort {
        attribute maintenanceScheduled : Boolean;
        attribute maintenanceComplete : Boolean;
    }

    part def TemperatureSensor {
        attribute sensorId : String;
        attribute measuredTemperature : Real;
        attribute samplingInterval : Real;
        port sensorOut : TemperatureSensorPort;
    }

    part def TemperatureMonitor {
        attribute maxAllowableTemperature : Real;
        attribute currentTemperature : Real;
        attribute isOverTemperature : Boolean;
        port monitorSensorIn : TemperatureSensorPort;
        port monitorControllerOut : ControllerPort;
    }

    part def MaintenanceScheduler {
        attribute maintenanceIntervalHours : Real;
        attribute lastMaintenanceTimeHours : Real;
        attribute nextMaintenanceTimeHours : Real;
        attribute maintenanceDurationHours : Real;
        attribute maxMaintenanceDurationHours : Real = 48.0;
        attribute maintenanceActive : Boolean;
        port schedulerPort : MaintenancePort;
    }

    part def VehicleController {
        attribute vehicleId : String;
        attribute currentStatus : VehicleHealthStatus;
        attribute activeWarning : WarningType;
        attribute operatingTimeHours : Real;
        port controllerIn : ControllerPort;
        port maintenanceIn : MaintenancePort;
    }

    part def VehicleHealthManagementSystem {
        attribute systemId : String;
        attribute systemName : String = "VehicleHealthMonitoringSystem";

        part sensor : TemperatureSensor {
            attribute sensorId : String = "TEMP_SENSOR_01";
            attribute samplingInterval : Real = 1.0;
        }

        part temperatureMonitor : TemperatureMonitor {
            attribute maxAllowableTemperature : Real = 100.0;
            attribute isOverTemperature : Boolean = false;
        }

        part maintenanceScheduler : MaintenanceScheduler {
            attribute maintenanceIntervalHours : Real = 500.0;
            attribute maxMaintenanceDurationHours : Real = 48.0;
            attribute maintenanceActive : Boolean = false;
        }

        part vehicleController : VehicleController {
            attribute vehicleId : String = "VEHICLE_01";
            attribute currentStatus : VehicleHealthStatus = VehicleHealthStatus::NORMAL;
            attribute activeWarning : WarningType = WarningType::NONE;
        }

        connect sensor.sensorOut to temperatureMonitor.monitorSensorIn;
        connect temperatureMonitor.monitorControllerOut to vehicleController.controllerIn;
        connect maintenanceScheduler.schedulerPort to vehicleController.maintenanceIn;
    }

    part vehicleSystem : VehicleHealthManagementSystem {
        attribute systemId : String = "VHMS_001";
    }
}