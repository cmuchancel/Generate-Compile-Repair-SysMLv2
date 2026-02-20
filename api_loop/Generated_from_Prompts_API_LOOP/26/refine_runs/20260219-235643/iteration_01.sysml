package AutomaticBrakingControl
{
    private import ScalarValues::*;

    enum def KeyPosition { Off; Start; Run; }
    enum def RoadAdhesion { Low; Medium; High; }
    enum def BrakeStatus { Released; Pressed; }

    port def BrakePressurePort { attribute pressure: Real; attribute status: BrakeStatus; }
    port def TractionPort { attribute adhesion: RoadAdhesion; }
    port def ControlPort { attribute targetPressure: Real; attribute modulationFrequency: Real; attribute active: Boolean; }
    port def ActuatorPort { attribute appliedPressure: Real; }

    part def BrakePedalSensor
    {
        port outPressure: BrakePressurePort;
        attribute brakePressure: Real;
        attribute brakeStatus: BrakeStatus;
    }

    part def TractionSensor
    {
        port outTraction: TractionPort;
        attribute adhesion: RoadAdhesion;
    }

    part def BrakeController
    {
        port inPressure: BrakePressurePort;
        port inTraction: TractionPort;
        port outControl: ControlPort;
        attribute targetPressure: Real;
        attribute modulationFrequency: Real;
        attribute active: Boolean;
        attribute monitoring: Boolean;
    }

    part def BrakeActuator
    {
        port inControl: ControlPort;
        port outApplied: ActuatorPort;
        attribute appliedPressure: Real;
    }

    part def AutomaticBrakingSystem
    {
        attribute keyPosition: KeyPosition;
        attribute active: Boolean;
        attribute monitoring: Boolean;
        attribute automaticallyActivated: Boolean;
        part pedalSensor: BrakePedalSensor;
        part tractionSensor: TractionSensor;
        part controller: BrakeController;
        part actuator: BrakeActuator;
        connect pedalSensor.outPressure to controller.inPressure;
        connect tractionSensor.outTraction to controller.inTraction;
        connect controller.outControl to actuator.inControl;
    }

    part def Vehicle
    {
        attribute keyPosition: KeyPosition;
        attribute brakePedalPressed: Boolean;
        part brakingSystem: AutomaticBrakingSystem;
    }

    part vehicle: Vehicle;
}