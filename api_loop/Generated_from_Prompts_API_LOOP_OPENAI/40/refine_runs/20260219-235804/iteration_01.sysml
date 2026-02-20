package VehicleHealthSystem
{
    private import ScalarValues::*;
    
    enum def VehicleStatus { normal; degraded; maintenance; }
    
    item def OverTempWarning;
    
    port def WarningPort
    {
        item warning : OverTempWarning;
    }
    
    part def VehicleHealthMonitor
    {
        attribute currentTemp : Real;
        attribute maxAllowTemp : Real;
        attribute status : VehicleStatus = normal;
        attribute maintenanceIntervalHours : Integer;
        attribute nextMaintenanceDueHours : Integer;
        attribute maintenanceElapsedHours : Integer;
        port warningOut : WarningPort;
    }
    
    part def VehicleController
    {
        port warningIn : WarningPort;
        attribute receivedWarning : Boolean;
    }
    
    part def VehicleSystem
    {
        part monitor : VehicleHealthMonitor;
        part controller : VehicleController;
        connect monitor.warningOut to controller.warningIn;
    }
}