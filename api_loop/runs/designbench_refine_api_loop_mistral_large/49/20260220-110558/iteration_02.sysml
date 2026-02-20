package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedUnit {
        enum MPH;
        enum KPH;
    }

    part def SpeedSensor {
        attribute currentSpeed : Real;
        attribute speedUnit : SpeedUnit = SpeedUnit::KPH;
        port speedOut : SpeedOutputPort;
    }

    part def CruiseController {
        port speedIn : SpeedInputPort;
        port setSpeedIn : SetSpeedInputPort;
        port fuelCommandOut : FuelCommandOutputPort;
    }

    part def EngineController {
        port fuelCommandIn : FuelCommandInputPort;
        port fuelCommandOut : FuelCommandOutputPort;
    }

    part def Engine {
        port fuelCommandIn : FuelCommandInputPort;
    }

    port def SpeedOutputPort {
        attribute speed : Real;
    }

    port def SpeedInputPort {
        attribute speed : Real;
    }

    port def SetSpeedInputPort {
        attribute setSpeed : Real;
    }

    port def FuelCommandOutputPort {
        attribute commandValue : Real;
    }

    port def FuelCommandInputPort {
        attribute commandValue : Real;
    }

    part vehicle {
        part speedSensor : SpeedSensor;
        part cruiseController : CruiseController;
        part engineController : EngineController;
        part engine : Engine;
    }

    connect vehicle.speedSensor.speedOut to vehicle.cruiseController.speedIn;
    connect vehicle.cruiseController.setSpeedIn to vehicle::CruiseControlSystem::SetSpeedInputPort;
    connect vehicle.cruiseController.fuelCommandOut to vehicle.engineController.fuelCommandIn;
    connect vehicle.engineController.fuelCommandOut to vehicle.engine.fuelCommandIn;
}