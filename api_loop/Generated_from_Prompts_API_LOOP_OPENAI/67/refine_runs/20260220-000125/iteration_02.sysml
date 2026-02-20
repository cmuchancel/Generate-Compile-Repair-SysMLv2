package VehicleMaintenance {
  private import ScalarValues::*;

  enum def VehicleStatus {
    enum normal;
    enum underMaintenance;
  }

  part def Vehicle {
    attribute scheduledMaintenanceTime : Real;
    attribute maintenanceInterval : Real;
    attribute maxAllowedTemperature : Real;
    attribute maintenanceTimeReached : Boolean;
    attribute healthStatus : VehicleStatus = VehicleStatus::normal;
    attribute actualMaintenanceStartTime : Real;
    attribute previousHealthyEndTime : Real;
    attribute maintenanceStartWithin2s : Boolean;
    attribute maintenanceDuration : Real;
    attribute maintenanceDurationWithin48h : Boolean;
    attribute maintenanceCompleted : Boolean;
    attribute nextMaintenanceTime : Real;
  }

  part def MaintenanceSystem {
    part vehicles : Vehicle[0..*];
  }
}