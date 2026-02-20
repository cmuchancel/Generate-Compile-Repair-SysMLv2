package VehicleMaintenanceSystem {
    private import ScalarValues::*;

    enum def VehicleStatus {
        enum healthy;
        enum underMaintenance;
    }

    enum def MaintenanceResult {
        enum pending;
        enum completed;
    }

    part def MaintenanceSchedule {
        attribute nextMaintenanceTime : Real;
        attribute maintenanceInterval : Real;
        attribute scheduledDuration : Real;
    }

    part def MaintenanceRecord {
        attribute actualStartTime : Real;
        attribute actualEndTime : Real;
        attribute result : MaintenanceResult;
    }

    part def Vehicle {
        attribute vehicleId : String;
        attribute maxAllowedTemperature : Real;
        attribute currentTemperature : Real;
        attribute currentStatus : VehicleStatus;
        attribute currentTime : Real;
        attribute healthyStateEndTime : Real;
        attribute maintenanceDurationHours : Real;
        attribute startDelaySeconds : Real;
        attribute isMaintenanceDue : Boolean;
        attribute startTimeValid : Boolean;
        attribute delayValid : Boolean;
        attribute durationValid : Boolean;

        part schedule : MaintenanceSchedule;
        part record : MaintenanceRecord;
    }

    part def MaintenanceController {
        attribute maxMaintenanceDurationHours : Real = 48.0;
        attribute maxStartDelaySeconds : Real = 2.0;

        part managedVehicle : Vehicle;
    }

    part def TemperatureSensor {
        attribute sensorId : String;
        attribute measuredTemperature : Real;
        attribute alertThreshold : Real;
        attribute alertTriggered : Boolean;
    }

    port def VehicleDataPort {
        attribute vehicleStatus : VehicleStatus;
        attribute temperature : Real;
        attribute nextMaintenance : Real;
    }

    port def ControllerPort {
        attribute commandIssued : Boolean;
        attribute targetVehicleId : String;
    }

    part def MaintenanceSystem {
        attribute systemId : String;
        attribute systemTime : Real;

        part controller : MaintenanceController;
        part sensor : TemperatureSensor;

        port dataOut : VehicleDataPort;
        port controlIn : ControllerPort;
    }

    part vehicleA : Vehicle {
        attribute vehicleId : String = "V001";
        attribute maxAllowedTemperature : Real = 90.0;
        attribute currentTemperature : Real = 75.0;
        attribute currentStatus : VehicleStatus = VehicleStatus::healthy;
        attribute currentTime : Real = 1000.0;
        attribute healthyStateEndTime : Real = 999.0;
        attribute maintenanceDurationHours : Real = 24.0;
        attribute startDelaySeconds : Real = 1.5;
        attribute isMaintenanceDue : Boolean = true;
        attribute startTimeValid : Boolean = true;
        attribute delayValid : Boolean = true;
        attribute durationValid : Boolean = true;

        part schedule : MaintenanceSchedule {
            attribute nextMaintenanceTime : Real = 1000.0;
            attribute maintenanceInterval : Real = 720.0;
            attribute scheduledDuration : Real = 24.0;
        }

        part record : MaintenanceRecord {
            attribute actualStartTime : Real = 1000.5;
            attribute actualEndTime : Real = 1024.5;
            attribute result : MaintenanceResult = MaintenanceResult::completed;
        }
    }

    part system1 : MaintenanceSystem {
        attribute systemId : String = "SYS001";
        attribute systemTime : Real = 1000.0;

        part controller : MaintenanceController {
            attribute maxMaintenanceDurationHours : Real = 48.0;
            attribute maxStartDelaySeconds : Real = 2.0;

            part managedVehicle : Vehicle {
                attribute vehicleId : String = "V001";
                attribute maxAllowedTemperature : Real = 90.0;
                attribute currentTemperature : Real = 75.0;
                attribute currentStatus : VehicleStatus = VehicleStatus::underMaintenance;
                attribute currentTime : Real = 1000.0;
                attribute healthyStateEndTime : Real = 999.0;
                attribute maintenanceDurationHours : Real = 24.0;
                attribute startDelaySeconds : Real = 1.5;
                attribute isMaintenanceDue : Boolean = true;
                attribute startTimeValid : Boolean = true;
                attribute delayValid : Boolean = true;
                attribute durationValid : Boolean = true;

                part schedule : MaintenanceSchedule {
                    attribute nextMaintenanceTime : Real = 1720.0;
                    attribute maintenanceInterval : Real = 720.0;
                    attribute scheduledDuration : Real = 24.0;
                }

                part record : MaintenanceRecord {
                    attribute actualStartTime : Real = 1000.5;
                    attribute actualEndTime : Real = 1024.5;
                    attribute result : MaintenanceResult = MaintenanceResult::completed;
                }
            }
        }

        part sensor : TemperatureSensor {
            attribute sensorId : String = "SENS001";
            attribute measuredTemperature : Real = 75.0;
            attribute alertThreshold : Real = 90.0;
            attribute alertTriggered : Boolean = false;
        }

        port dataOut : VehicleDataPort;
        port controlIn : ControllerPort;
    }

    connect system1.dataOut to system1.controlIn;
}