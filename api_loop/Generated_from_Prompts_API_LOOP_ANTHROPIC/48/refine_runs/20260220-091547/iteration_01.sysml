package CruiseControl {
    private import ScalarValues::*;

    enum def SpeedCommandType {
        enum SET;
        enum CANCEL;
        enum RESUME;
    }

    item def SetSpeedItem {
        attribute targetSpeed : Real;
        attribute command : SpeedCommandType;
    }

    item def ActualSpeedItem {
        attribute actualSpeed : Real;
    }

    item def FuelControlItem {
        attribute fuelAdjustment : Real;
    }

    port def DriverInputPort {
        out item setSpeedSignal : SetSpeedItem;
    }

    port def DriverInputReceivePort {
        in item setSpeedSignal : SetSpeedItem;
    }

    port def SpeedSensorOutputPort {
        out item speedReading : ActualSpeedItem;
    }

    port def SpeedSensorReceivePort {
        in item speedReading : ActualSpeedItem;
    }

    port def FuelControlOutputPort {
        out item fuelCommand : FuelControlItem;
    }

    port def FuelControlReceivePort {
        in item fuelCommand : FuelControlItem;
    }

    part def CruiseControlInterface {
        attribute driverSetSpeed : Real;
        attribute commandType : SpeedCommandType;
        port driverOut : DriverInputPort;
    }

    part def SpeedSensor {
        attribute measuredSpeed : Real;
        port sensorOut : SpeedSensorOutputPort;
    }

    part def CruiseController {
        attribute targetSpeed : Real;
        attribute currentSpeed : Real;
        attribute speedError : Real;
        port controllerDriverIn : DriverInputReceivePort;
        port controllerSensorIn : SpeedSensorReceivePort;
        port controllerFuelOut : FuelControlOutputPort;
    }

    part def Engine {
        attribute fuelLevel : Real;
        attribute powerOutput : Real;
        port engineFuelIn : FuelControlReceivePort;
    }

    part def CruiseControlSystem {
        part interface : CruiseControlInterface;
        part sensor : SpeedSensor;
        part controller : CruiseController;
        part engine : Engine;

        connect interface.driverOut to controller.controllerDriverIn;
        connect sensor.sensorOut to controller.controllerSensorIn;
        connect controller.controllerFuelOut to engine.engineFuelIn;
    }
}