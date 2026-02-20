package VehicleMaintenance
  private import ScalarValues::*;

  enum def VehicleStatus {
    enum normal;
    enum underMaintenance;
  }

  part def Vehicle {
    attribute scheduledMaintenanceTime : Time;
    attribute maintenanceInterval : Duration;
    attribute maxAllowedTemperature : Real;
    attribute maintenanceTimeReached : Boolean;
    attribute healthStatus : VehicleStatus = normal;
    attribute actualMaintenanceStartTime : Time;
    attribute previousHealthyEndTime : Time;
    attribute maintenanceStartWithin2s : Boolean;
    attribute maintenanceDuration : Duration;
    attribute maintenanceDurationWithin48h : Boolean;
    attribute maintenanceCompleted : Boolean;
    attribute nextMaintenanceTime : Time;
  }

  part def MaintenanceSystem {
    part vehicles : Vehicle[0..*];
  }
end package