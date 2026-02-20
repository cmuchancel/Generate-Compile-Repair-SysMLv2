package VehicleMaintenanceSystem {
    private import ScalarValues::*;
    
    enum def HealthStatus {
        enum healthy;
        enum under_maintenance;
    }
    
    part def Vehicle {
        attribute maintenanceInterval : Real;
        attribute nextMaintenanceTime : Real;
        attribute maxAllowedTemperature : Real;
        attribute currentHealth : HealthStatus;
        attribute lastHealthyEndTime : Real;
        attribute maintenanceStartTime : Real;
        attribute maintenanceEndTime : Real;
    }
    
    part def MaintenanceSystem {
        part monitoredVehicles : Vehicle[1..*];
    }
    
    part SystemInstance {
        part controller : MaintenanceSystem;
        part vehicle1 : Vehicle;
        part vehicle2 : Vehicle;
        
        connect controller.monitoredVehicles to vehicle1;
        connect controller.monitoredVehicles to vehicle2;
    }
}