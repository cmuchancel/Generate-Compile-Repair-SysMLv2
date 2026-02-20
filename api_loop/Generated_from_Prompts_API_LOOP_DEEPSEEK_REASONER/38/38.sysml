package VehicleStateManagement {
    private import ScalarValues::*;

    enum def VehicleState {
        enum shutdown;
        enum start_up;
        enum running;
    }

    enum def Signal {
        enum vehicle_start;
        enum vehicle_started;
        enum vehicle_shutdown;
    }

    part def Vehicle {
        port command : Signal;
        attribute status : VehicleState = VehicleState::shutdown;
    }

    part def User {
        port command : Signal;
    }

    part vehicle : Vehicle;
    part user : User;

    connect user.command to vehicle.command;
}