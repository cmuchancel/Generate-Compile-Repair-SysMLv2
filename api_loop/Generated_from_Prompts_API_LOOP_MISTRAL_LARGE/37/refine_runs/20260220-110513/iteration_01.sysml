package VehiclePowerManagement {
    private import ScalarValues::*;

    enum def PowerState {
        enum off;
        enum startPhase;
        enum poweredOn;
    }

    enum def ControlSignal {
        enum startCommand;
        enum startedSignal;
        enum shutdownCommand;
    }

    part def Vehicle {
        attribute powerState : PowerState = PowerState::off;
        attribute isDiagnosing : Boolean = false;
        attribute isPowerSupplied : Boolean = false;
        attribute isBrakeEngaged : Boolean = false;

        port controlIn : ControlSignalPort;
        port powerOut : PowerSupplyPort;
    }

    part def ControlUnit {
        port controlOut : ControlSignalPort;
    }

    part def FunctionalUnit {
        port powerIn : PowerSupplyPort;
    }

    port def ControlSignalPort {
        attribute signal : ControlSignal;
    }

    port def PowerSupplyPort {
        attribute isPowered : Boolean;
    }

    item def ControlSignalItem {
        attribute signal : ControlSignal;
    }

    part vehicle : Vehicle;
    part controlUnit : ControlUnit;
    part functionalUnit : FunctionalUnit;

    connect controlUnit.controlOut to vehicle.controlIn;
    connect vehicle.powerOut to functionalUnit.powerIn;
}