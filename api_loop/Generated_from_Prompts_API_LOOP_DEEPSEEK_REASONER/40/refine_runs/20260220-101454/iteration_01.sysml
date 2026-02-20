package VehicleHealthMonitoringSystem {
    private import ScalarValues::*;

    enum def HealthStatus {
        enum normal;
        enum degraded;
        enum maintenance;
    }

    part def VehicleController;

    part def VehicleHealthMonitor {
        port def warningOutput;
        port warningPort : warningOutput;

        attribute currentTemperature : Real;
        attribute maxAllowableTemperature : Real = 100.0;
        attribute healthStatus : HealthStatus = HealthStatus::normal;
        attribute nextMaintenanceTime : Real;
        attribute maintenanceInterval : Real = 720.0;
        attribute maintenanceStartTime : Real;
        attribute maintenanceDurationThreshold : Real = 48.0;
    }

    part vehicleController : VehicleController;
    part vehicleMonitor : VehicleHealthMonitor;

    connect vehicleMonitor.warningPort to vehicleController;
}