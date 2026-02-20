package SensorSystem {
    private import ScalarValues::*;
    
    enum def SensorStatus {
        enum Ready,
        enum Updating,
        enum Updated,
        enum Error
    }
    
    part def Sensor {
        attribute id : String;
        attribute status : SensorStatus;
        attribute lastUpdateTime : Integer = 0;
    }
    
    part def SensorManager {
        part sensors : Sensor[0..*];
        attribute updateInterval : Integer = 1000;
        
        port def SensorMonitor;
        port monitor : SensorMonitor;
        
        port def UpdateController;
        port controller : UpdateController;
    }
    
    part system : SensorManager;
    
    connect system.monitor to system.sensors;
    connect system.controller to system.sensors;
}