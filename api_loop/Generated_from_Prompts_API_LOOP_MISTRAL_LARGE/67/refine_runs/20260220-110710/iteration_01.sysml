package VehicleMaintenanceSystem {
    private import ScalarValues::*;

    enum def MaintenanceStatus {
        enum healthy;
        enum under_maintenance;
    }

    part def Vehicle {
        attribute id : String;
        attribute nextMaintenanceTime : Real;
        attribute maintenanceInterval : Real;
        attribute maxAllowedTemperature : Real;
        attribute status : MaintenanceStatus = MaintenanceStatus::healthy;
        attribute lastHealthyEndTime : Real;
        attribute actualMaintenanceStartTime : Real;
        attribute maintenanceDuration : Real;
    }

    part def MaintenanceScheduler {
        port schedulePort : SchedulePort;
        port monitorPort : MonitorPort;
    }

    port def SchedulePort {
    }

    port def MonitorPort {
    }

    item def MaintenanceEvent {
        attribute scheduledTime : Real;
        attribute actualStartTime : Real;
        attribute duration : Real;
    }

    part vehicle : Vehicle;
    part scheduler : MaintenanceScheduler;

    connect scheduler.schedulePort to vehicle;
    connect scheduler.monitorPort to vehicle;
}