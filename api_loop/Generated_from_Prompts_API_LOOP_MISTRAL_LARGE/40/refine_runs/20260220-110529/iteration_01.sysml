package VehicleHealthMonitoring {
    private import ScalarValues::*;

    enum def HealthStatus {
        enum normal;
        enum degraded;
        enum maintenance;
    }

    part def VehicleController;

    part def TemperatureSensor {
        attribute currentTemperature : Real;
    }

    part def HealthMonitor {
        attribute maxAllowableTemp : Real = 100.0;
        attribute maintenanceInterval : Real = 30.0; // days
        attribute nextMaintenanceTime : Real;
        attribute healthStatus : HealthStatus = HealthStatus::normal;
        attribute maintenanceDuration : Real = 48.0; // hours
        attribute maintenanceStartTime : Real;

        port tempInput : TemperatureSensor;
        port controllerOutput : VehicleController;
    }

    part def VehicleSystem {
        part temperatureSensor : TemperatureSensor;
        part healthMonitor : HealthMonitor;
        part vehicleController : VehicleController;

        connect temperatureSensor to healthMonitor.tempInput;
        connect healthMonitor.controllerOutput to vehicleController;
    }
}