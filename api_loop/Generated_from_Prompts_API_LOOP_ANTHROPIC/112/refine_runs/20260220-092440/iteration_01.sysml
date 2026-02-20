package VehicleManagementSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        enum Off;
        enum Starting;
        enum Running;
        enum Degraded;
        enum Maintenance;
        enum Normal;
    }

    enum def ControllerState {
        enum Idle;
        enum Active;
        enum Shutdown;
    }

    enum def HealthStatus {
        enum Healthy;
        enum Overtemperature;
        enum Degraded;
    }

    item def StartSignal;
    item def VehicleOnSignal;
    item def ShutdownSignal;
    item def RestoreNormalSignal;
    item def OvertemperatureSignal;
    item def BrakePedalSignal;
    item def ParkingBrakeCommand;
    item def PowerSupplyCommand;
    item def SelfCheckCommand;

    port def SignalInputPort {
        attribute signalPresent : Boolean;
    }

    port def SignalOutputPort {
        attribute signalPresent : Boolean;
    }

    port def TemperaturePort {
        attribute temperatureValue : Real;
    }

    port def StatePort {
        attribute currentState : VehicleState;
    }

    part def HealthMonitor {
        attribute currentTemperature : Real;
        attribute temperatureThreshold : Real = 120.0;
        attribute healthStatus : HealthStatus;
        attribute overtemperatureDetected : Boolean;

        port temperatureOut : TemperaturePort;
        port overtemperatureSignalOut : SignalOutputPort;
    }

    part def ParkingBrakeSystem {
        attribute brakeEngaged : Boolean;
        port brakeCommandIn : SignalInputPort;
    }

    part def PowerSupplySystem {
        attribute powerOn : Boolean;
        port powerCommandIn : SignalInputPort;
    }

    part def SelfCheckSystem {
        attribute checkComplete : Boolean;
        attribute checkPassed : Boolean;
        port selfCheckCommandIn : SignalInputPort;
    }

    part def VehicleController {
        attribute controllerState : ControllerState;
        attribute vehicleMainState : VehicleState;

        port startSignalIn : SignalInputPort;
        port shutdownSignalIn : SignalInputPort;
        port overtemperatureSignalIn : SignalInputPort;
        port restoreNormalSignalIn : SignalInputPort;
        port vehicleStateOut : StatePort;
        port parkingBrakeCommandOut : SignalOutputPort;
        port powerCommandOut : SignalOutputPort;
        port selfCheckCommandOut : SignalOutputPort;
    }

    part def VehicleStartupManager {
        attribute brakePedalPressed : Boolean;
        attribute startSignalReceived : Boolean;
        attribute vehicleOnSignalReceived : Boolean;
        attribute currentState : VehicleState;
        attribute canTransitionToStarting : Boolean;

        port brakePedalIn : SignalInputPort;
        port startSignalIn : SignalInputPort;
        port vehicleOnSignalIn : SignalInputPort;
        port startupStateOut : StatePort;
    }

    part def VehicleOperatingSystem {
        attribute operatingState : VehicleState;
        attribute selfCheckActive : Boolean;
        attribute powerSupplyActive : Boolean;
        attribute parkingBrakeActive : Boolean;

        port operatingStateIn : StatePort;
        port operatingStateOut : StatePort;
        port parkingBrakeOut : SignalOutputPort;
        port powerSupplyOut : SignalOutputPort;
        port selfCheckOut : SignalOutputPort;
    }

    part def VehicleA {
        attribute vehicleId : String;
        attribute currentState : VehicleState;
        attribute healthStatus : HealthStatus;
        attribute temperature : Real;
        attribute temperatureThreshold : Real = 120.0;
        attribute brakePedalPressed : Boolean;
        attribute startSignalReceived : Boolean;
        attribute vehicleOnSignalReceived : Boolean;
        attribute overtemperatureDetected : Boolean;
        attribute parkingBrakeEngaged : Boolean;
        attribute powerSupplyOn : Boolean;
        attribute selfCheckPassed : Boolean;
        attribute maintenanceMode : Boolean;
        attribute degradedMode : Boolean;

        port vehicleStartSignalIn : SignalInputPort;
        port vehicleBrakePedalIn : SignalInputPort;
        port vehicleOnSignalIn : SignalInputPort;
        port vehicleShutdownSignalIn : SignalInputPort;
        port vehicleRestoreNormalIn : SignalInputPort;
        port vehicleTemperatureOut : TemperaturePort;
        port vehicleOvertemperatureOut : SignalOutputPort;
        port vehicleStateOut : StatePort;

        part healthMonitor : HealthMonitor;
        part parkingBrake : ParkingBrakeSystem;
        part powerSupply : PowerSupplySystem;
        part selfCheck : SelfCheckSystem;
        part startupManager : VehicleStartupManager;
        part operatingSystem : VehicleOperatingSystem;
        part controller : VehicleController;

        connect startupManager.brakePedalIn to vehicleBrakePedalIn;
        connect startupManager.startSignalIn to vehicleStartSignalIn;
        connect startupManager.vehicleOnSignalIn to vehicleOnSignalIn;
        connect startupManager.startupStateOut to operatingSystem.operatingStateIn;
        connect operatingSystem.parkingBrakeOut to parkingBrake.brakeCommandIn;
        connect operatingSystem.powerSupplyOut to powerSupply.powerCommandIn;
        connect operatingSystem.selfCheckOut to selfCheck.selfCheckCommandIn;
        connect healthMonitor.temperatureOut to vehicleTemperatureOut;
        connect healthMonitor.overtemperatureSignalOut to controller.overtemperatureSignalIn;
        connect healthMonitor.overtemperatureSignalOut to vehicleOvertemperatureOut;
        connect controller.startSignalIn to vehicleStartSignalIn;
        connect controller.shutdownSignalIn to vehicleShutdownSignalIn;
        connect controller.restoreNormalSignalIn to vehicleRestoreNormalIn;
        connect controller.parkingBrakeCommandOut to parkingBrake.brakeCommandIn;
        connect controller.powerCommandOut to powerSupply.powerCommandIn;
        connect controller.selfCheckCommandOut to selfCheck.selfCheckCommandIn;
        connect controller.vehicleStateOut to vehicleStateOut;
        connect operatingSystem.operatingStateOut to vehicleStateOut;
    }
}