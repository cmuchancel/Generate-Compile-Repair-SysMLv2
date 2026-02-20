package AutomaticBrakingSystem {
    private import ScalarValues::*;

    enum def KeyPosition {
        enum OFF;
        enum ACC;
        enum ON;
        enum START;
    }

    enum def BrakePedalStatus {
        enum RELEASED;
        enum PRESSED;
    }

    item def BrakePressure {
        attribute value : Real;
    }

    item def RoadAdhesion {
        attribute coefficient : Real;
    }

    item def ModulationFrequency {
        attribute value : Real;
    }

    port def SensorSignal {
        attribute brakePressure : BrakePressure;
        attribute roadAdhesion : RoadAdhesion;
    }

    port def ControlSignal {
        attribute targetPressure : BrakePressure;
        attribute modulationFrequency : ModulationFrequency;
    }

    port def PowerSignal {
        attribute isActive : Boolean;
    }

    part def BrakePedalSensor {
        port pedalSignal : SensorSignal;
        attribute status : BrakePedalStatus;
    }

    part def TractionSensor {
        port tractionSignal : SensorSignal;
    }

    part def ControlUnit {
        port sensorInput : SensorSignal;
        port controlOutput : ControlSignal;
        port powerInput : PowerSignal;
    }

    part def BrakeActuator {
        port controlInput : ControlSignal;
    }

    part def IgnitionSystem {
        port powerOutput : PowerSignal;
        attribute keyPosition : KeyPosition;
    }

    part def AutomaticBrakingSystem {
        part brakePedalSensor : BrakePedalSensor;
        part tractionSensor : TractionSensor;
        part controlUnit : ControlUnit;
        part brakeActuator : BrakeActuator;
        part ignitionSystem : IgnitionSystem;

        connect brakePedalSensor.pedalSignal to controlUnit.sensorInput;
        connect tractionSensor.tractionSignal to controlUnit.sensorInput;
        connect controlUnit.controlOutput to brakeActuator.controlInput;
        connect ignitionSystem.powerOutput to controlUnit.powerInput;
    }
}