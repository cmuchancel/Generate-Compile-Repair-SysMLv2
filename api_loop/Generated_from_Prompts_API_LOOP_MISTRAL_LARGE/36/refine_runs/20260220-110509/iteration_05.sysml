package VehicleOperationalStatus {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::off;
    }

    port def VehicleStartPort {
        in item startSignal : VehicleStartSignal;
    }

    port def VehicleOnPort {
        in item onSignal : VehicleOnSignal;
    }

    port def VehicleOffPort {
        in item offSignal : VehicleOffSignal;
    }

    item def VehicleStartSignal;
    item def VehicleOnSignal;
    item def VehicleOffSignal;

    part vehicle : Vehicle;
    part startPort : VehicleStartPort;
    part onPort : VehicleOnPort;
    part offPort : VehicleOffPort;

    connect startPort to vehicle;
    connect onPort to vehicle;
    connect offPort to vehicle;
}