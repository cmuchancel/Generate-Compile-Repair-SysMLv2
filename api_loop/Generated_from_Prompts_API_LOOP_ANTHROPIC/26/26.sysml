package AutomaticBrakingControlSystem {
    private import ScalarValues::*;

    enum def SystemStatus {
        enum INACTIVE;
        enum ACTIVE;
        enum MONITORING;
        enum REGULATING;
    }

    enum def BrakingMode {
        enum NORMAL;
        enum ABS_ACTIVE;
        enum TRACTION_CONTROL;
    }

    enum def RoadCondition {
        enum DRY;
        enum WET;
        enum ICY;
        enum LOOSE;
    }

    item def BrakePressureSignal {
        attribute pressureValue : Real;
        attribute timestamp : Real;
    }

    item def TractionSignal {
        attribute adhesionCoefficient : Real;
        attribute roadCondition : RoadCondition;
        attribute timestamp : Real;
    }

    item def BrakeCommand {
        attribute targetPressure : Real;
        attribute modulationFrequency : Real;
        attribute mode : BrakingMode;
    }

    port def KeyInputPort {
        attribute keyPosition : Boolean;
    }

    port def BrakePedalPort {
        attribute pedalDepressed : Boolean;
        attribute pedalPressure : Real;
    }

    port def TractionSensorPort {
        attribute adhesionCoefficient : Real;
        attribute roadCondition : RoadCondition;
    }

    port def BrakeActuatorPort {
        attribute commandedPressure : Real;
        attribute modulationFrequency : Real;
    }

    port def SystemStatusPort {
        attribute currentStatus : SystemStatus;
        attribute currentMode : BrakingMode;
    }

    part def IgnitionMonitor {
        attribute systemActive : Boolean = false;
        attribute keyEngaged : Boolean = false;

        port keyIn : KeyInputPort;
        port statusOut : SystemStatusPort;
    }

    part def BrakePedalSensor {
        attribute detectedPressure : Real = 0.0;
        attribute pedalEngaged : Boolean = false;
        attribute sensorOperational : Boolean = true;

        port pedalIn : BrakePedalPort;
        port pressureOut : BrakePedalPort;
    }

    part def TractionMonitor {
        attribute currentAdhesion : Real = 1.0;
        attribute currentRoadCondition : RoadCondition;
        attribute tractionVariation : Real = 0.0;
        attribute monitoringActive : Boolean = false;

        port tractionIn : TractionSensorPort;
        port tractionDataOut : TractionSensorPort;
    }

    part def BrakePressureController {
        attribute targetPressure : Real = 0.0;
        attribute currentPressure : Real = 0.0;
        attribute modulationFrequency : Real = 0.0;
        attribute controllerActive : Boolean = false;
        attribute currentMode : BrakingMode;

        port pedalDataIn : BrakePedalPort;
        port tractionDataIn : TractionSensorPort;
        port brakeCommandOut : BrakeActuatorPort;
        port statusIn : SystemStatusPort;
    }

    part def BrakeActuator {
        attribute appliedPressure : Real = 0.0;
        attribute modulationFrequency : Real = 0.0;
        attribute actuatorOperational : Boolean = true;
        attribute brakeEngaged : Boolean = false;

        port commandIn : BrakeActuatorPort;
        port actuatorStatus : BrakeActuatorPort;
    }

    part def SafetyMonitor {
        attribute safetyCheckActive : Boolean = false;
        attribute systemHealthy : Boolean = true;
        attribute interventionRequired : Boolean = false;

        port statusIn : SystemStatusPort;
        port brakeStatusIn : BrakeActuatorPort;
        port safetyCommandOut : BrakeActuatorPort;
    }

    part def AutomaticBrakingSystem {
        attribute systemStatus : SystemStatus;
        attribute brakingMode : BrakingMode;
        attribute continuousMonitoring : Boolean = true;
        attribute userInterventionRequired : Boolean = false;
        attribute optimalBrakingMaintained : Boolean = true;

        part ignitionMonitor : IgnitionMonitor;
        part brakePedalSensor : BrakePedalSensor;
        part tractionMonitor : TractionMonitor;
        part pressureController : BrakePressureController;
        part brakeActuator : BrakeActuator;
        part safetyMonitor : SafetyMonitor;

        connect ignitionMonitor.statusOut to pressureController.statusIn;
        connect brakePedalSensor.pressureOut to pressureController.pedalDataIn;
        connect tractionMonitor.tractionDataOut to pressureController.tractionDataIn;
        connect pressureController.brakeCommandOut to brakeActuator.commandIn;
        connect brakeActuator.actuatorStatus to safetyMonitor.brakeStatusIn;
        connect ignitionMonitor.statusOut to safetyMonitor.statusIn;
        connect safetyMonitor.safetyCommandOut to brakeActuator.commandIn;
    }
}