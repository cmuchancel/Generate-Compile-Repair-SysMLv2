package CruiseControlSystem {
    private import ScalarValues::*;
    
    item def SetSpeedCommand {
        attribute speed : Real;
    }
    
    item def VehicleSpeed {
        attribute speed : Real;
    }
    
    item def ThrottleCommand {
        attribute throttle : Real;
    }
    
    port def CommandPort {
        attribute command : SetSpeedCommand;
    }
    
    port def SpeedPort {
        attribute speed : VehicleSpeed;
    }
    
    port def ThrottlePort {
        attribute throttleCommand : ThrottleCommand;
    }
    
    part def Driver {
        port commandOut : CommandPort;
    }
    
    part def Speedometer {
        port speedOut : SpeedPort;
    }
    
    part def CruiseController {
        port commandIn : CommandPort;
        port speedIn : SpeedPort;
        port throttleOut : ThrottlePort;
    }
    
    part def Engine {
        port throttleIn : ThrottlePort;
    }
    
    part def Vehicle {
        part driver : Driver;
        part speedometer : Speedometer;
        part controller : CruiseController;
        part engine : Engine;
        
        connect driver.commandOut to controller.commandIn;
        connect speedometer.speedOut to controller.speedIn;
        connect controller.throttleOut to engine.throttleIn;
    }
}