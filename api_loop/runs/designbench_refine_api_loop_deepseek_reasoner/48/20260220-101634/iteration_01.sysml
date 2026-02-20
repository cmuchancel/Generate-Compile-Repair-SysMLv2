package CruiseControlSystem {
    private import ScalarValues::*;
    
    item def Speed {
        attribute value : Real;
    }
    
    item def FuelCommand {
        attribute value : Real;
    }
    
    port def SetSpeedPort {
        attribute targetSpeed : Speed;
    }
    
    port def SpeedSensorPort {
        attribute currentSpeed : Speed;
    }
    
    port def FuelControlPort {
        attribute command : FuelCommand;
    }
    
    part def CruiseController {
        port setSpeedIn : SetSpeedPort;
        port speedSensorIn : SpeedSensorPort;
        port fuelControlOut : FuelControlPort;
    }
    
    part def CruiseInterface {
        port setSpeedOut : SetSpeedPort;
    }
    
    part def SpeedSensor {
        port speedOut : SpeedSensorPort;
    }
    
    part def Engine {
        port fuelControlIn : FuelControlPort;
    }
    
    part def Vehicle {
        part controller : CruiseController;
        part interface : CruiseInterface;
        part sensor : SpeedSensor;
        part engine : Engine;
        
        connect interface.setSpeedOut to controller.setSpeedIn;
        connect sensor.speedOut to controller.speedSensorIn;
        connect controller.fuelControlOut to engine.fuelControlIn;
    }
}