package VehiclePowerManagement {
    private import ScalarValues::*;

    enum def VehicleState {
        enum Off;
        enum StartPhase;
        enum PoweredOn;
    }

    part def VehicleSystem {
        attribute currentState : VehicleState = VehicleState::Off;
        port startCommandIn : VehicleStartCommand;
        port startedSignalIn : VehicleStartedSignal;
        port shutdownCommandIn : VehicleShutdownCommand;
        port selfDiagnosisControl : SelfDiagnosisCommand;
        port powerSupplyControl : PowerSupplyCommand;
        port parkingBrakeControl : ParkingBrakeCommand;
    }

    part def SelfDiagnosisUnit {
        port control : SelfDiagnosisCommand;
    }

    part def PowerSupplyUnit {
        port control : PowerSupplyCommand;
    }

    part def ParkingBrake {
        port control : ParkingBrakeCommand;
        attribute isEngaged : Boolean = false;
    }

    item def VehicleStartCommand {}
    item def VehicleStartedSignal {}
    item def VehicleShutdownCommand {}
    item def SelfDiagnosisCommand {}
    item def PowerSupplyCommand {}
    item def ParkingBrakeCommand {}

    part vehicleSystem : VehicleSystem;
    part selfDiagnosisUnit : SelfDiagnosisUnit;
    part powerSupplyUnit : PowerSupplyUnit;
    part parkingBrake : ParkingBrake;

    connect vehicleSystem.selfDiagnosisControl to selfDiagnosisUnit.control;
    connect vehicleSystem.powerSupplyControl to powerSupplyUnit.control;
    connect vehicleSystem.parkingBrakeControl to parkingBrake.control;
}