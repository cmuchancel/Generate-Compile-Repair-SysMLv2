package VehicleOperationalStatus {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::off;
        port startPort : VehicleStartPort;
        port onPort : VehicleOnPort;
        port offPort : VehicleOffPort;
    }

    port def VehicleStartPort;
    port def VehicleOnPort;
    port def VehicleOffPort;

    item def VehicleStartSignal;
    item def VehicleOnSignal;
    item def VehicleOffSignal;

    part vehicle : Vehicle;

    connect vehicle.startPort to :>>VehicleStartSignal;
    connect vehicle.onPort to :>>VehicleOnSignal;
    connect vehicle.offPort to :>>VehicleOffSignal;
}