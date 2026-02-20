package VehiclePowerManagement {
    private import ScalarValues::*;

    enum def PowerState {
        enum OFF;
        enum STARTING;
        enum POWERED_ON;
    }

    enum def DiagnosticState {
        enum IDLE;
        enum RUNNING;
        enum COMPLETE;
    }

    enum def BrakeState {
        enum RELEASED;
        enum ENGAGED;
    }

    enum def CommandType {
        enum START_COMMAND;
        enum STOP_COMMAND;
        enum VEHICLE_STARTED_SIGNAL;
    }

    part def ControlSignal {
        attribute command : CommandType;
        attribute isActive : Boolean;
    }

    part def PowerSupply {
        attribute isOn : Boolean;
        attribute voltageLevel : Real;
    }

    part def SelfDiagnostics {
        attribute diagnosticState : DiagnosticState;
        attribute diagnosticResult : Boolean;
    }

    part def ParkingBrake {
        attribute brakeState : BrakeState;
        attribute isEngaged : Boolean;
    }

    part def FunctionalUnit {
        attribute unitName : String;
        attribute isPowered : Boolean;
        attribute isOperational : Boolean;
    }

    part def PowerStateController {
        attribute currentState : PowerState;
        attribute previousState : PowerState;
        attribute stateChangeAllowed : Boolean;
    }

    port def ControlPort {
        attribute commandReceived : CommandType;
        attribute signalActive : Boolean;
    }

    port def PowerPort {
        attribute powerAvailable : Boolean;
        attribute voltage : Real;
    }

    port def StatusPort {
        attribute statusValue : Boolean;
        attribute stateInfo : String;
    }

    part def VehicleStartModule {
        attribute startInitiated : Boolean;
        attribute startConfirmed : Boolean;
        attribute startPhaseActive : Boolean;
        port startControlPort : ControlPort;
        port startPowerPort : PowerPort;
        port startStatusPort : StatusPort;
    }

    part def VehicleShutdownModule {
        attribute shutdownInitiated : Boolean;
        attribute shutdownComplete : Boolean;
        attribute safetyOperationsComplete : Boolean;
        port shutdownControlPort : ControlPort;
        port shutdownStatusPort : StatusPort;
    }

    part def VehiclePowerSystem {
        attribute systemName : String;
        attribute overallPowerState : PowerState;
        attribute selfDiagnosticsEnabled : Boolean;
        attribute safeShutdownEnabled : Boolean;

        part powerStateController : PowerStateController;
        part powerSupply : PowerSupply;
        part selfDiagnostics : SelfDiagnostics;
        part parkingBrake : ParkingBrake;
        part startModule : VehicleStartModule;
        part shutdownModule : VehicleShutdownModule;

        part engineUnit : FunctionalUnit;
        part transmissionUnit : FunctionalUnit;
        part electricalUnit : FunctionalUnit;
        part safetyUnit : FunctionalUnit;

        part userControlSignal : ControlSignal;
        part vehicleStartedSignal : ControlSignal;

        port vehicleControlPort : ControlPort;
        port vehiclePowerPort : PowerPort;
        port vehicleStatusPort : StatusPort;

        connect vehicleControlPort to startModule.startControlPort;
        connect vehicleControlPort to shutdownModule.shutdownControlPort;
        connect startModule.startPowerPort to vehiclePowerPort;
        connect startModule.startStatusPort to vehicleStatusPort;
        connect shutdownModule.shutdownStatusPort to vehicleStatusPort;
    }

    item def PowerStatusReport {
        attribute reportId : String;
        attribute currentPowerState : PowerState;
        attribute diagnosticState : DiagnosticState;
        attribute brakeState : BrakeState;
        attribute allUnitsOperational : Boolean;
        attribute safetyConfirmed : Boolean;
    }

    item def StartCommandItem {
        attribute commandId : String;
        attribute commandType : CommandType;
        attribute timestamp : String;
        attribute userInitiated : Boolean;
    }

    item def ShutdownCommandItem {
        attribute commandId : String;
        attribute commandType : CommandType;
        attribute timestamp : String;
        attribute safetyCheckRequired : Boolean;
    }

    part vehiclePowerSystem : VehiclePowerSystem {
        attribute systemName = "VehiclePowerManagementSystem";
        attribute overallPowerState = PowerState::OFF;
        attribute selfDiagnosticsEnabled = true;
        attribute safeShutdownEnabled = true;
    }
}