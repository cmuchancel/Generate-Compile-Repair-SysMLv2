package CruiseControl {
    private import ScalarValues::*;

    item def SetSpeed {
        attribute value : Real;
    }

    item def ActualSpeed {
        attribute value : Real;
    }

    item def FuelCommand {
        attribute value : Real;
    }

    port def SetSpeedOutPort {
        out item setSpeed : SetSpeed;
    }

    port def SetSpeedInPort {
        in item setSpeed : SetSpeed;
    }

    port def ActualSpeedOutPort {
        out item actualSpeed : ActualSpeed;
    }

    port def ActualSpeedInPort {
        in item actualSpeed : ActualSpeed;
    }

    port def FuelCommandOutPort {
        out item fuelCommand : FuelCommand;
    }

    port def FuelCommandInPort {
        in item fuelCommand : FuelCommand;
    }

    part def DriverInterface {
        port setSpeedOut : SetSpeedOutPort;
    }

    part def CruiseController {
        port setSpeedIn : SetSpeedInPort;
        port actualSpeedIn : ActualSpeedInPort;
        port fuelCommandOut : FuelCommandOutPort;
    }

    part def SpeedSensor {
        port actualSpeedOut : ActualSpeedOutPort;
    }

    part def Engine {
        port fuelCommandIn : FuelCommandInPort;
    }

    part def CruiseControlSystem {
        part driverInterface : DriverInterface;
        part controller : CruiseController;
        part speedSensor : SpeedSensor;
        part engine : Engine;

        connect driverInterface.setSpeedOut to controller.setSpeedIn;
        connect speedSensor.actualSpeedOut to controller.actualSpeedIn;
        connect controller.fuelCommandOut to engine.fuelCommandIn;
    }
}