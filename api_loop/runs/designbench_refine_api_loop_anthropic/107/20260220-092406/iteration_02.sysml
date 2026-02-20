package VehicleTrailerHitchSystem {
    private import ScalarValues::*;

    enum def ConnectionState {
        enum CONNECTED;
        enum DISCONNECTED;
    }

    item def PowerSignal;

    port def VehiclePowerPort {
        out item power : PowerSignal;
    }

    port def TrailerPowerPort {
        in item power : PowerSignal;
    }

    part def VehicleFrame {
        attribute frameId : String;
    }

    part def TrailerFrame {
        attribute frameId : String;
    }

    part def HitchBall {
        attribute diameter : Real;
        attribute material : String;
    }

    part def TrailerCoupler {
        attribute diameter : Real;
        attribute material : String;
    }

    part def HitchMechanism {
        attribute connectionState : ConnectionState;
        port vehicleSide : VehiclePowerPort;
        port trailerSide : TrailerPowerPort;
        part hitchBall : HitchBall;
        part trailerCoupler : TrailerCoupler;
    }

    part def Vehicle {
        attribute vehicleId : String;
        part vehicleFrame : VehicleFrame;
        part hitch : HitchMechanism;
        port powerOut : VehiclePowerPort;
    }

    part def Trailer {
        attribute trailerId : String;
        part trailerFrame : TrailerFrame;
        port powerIn : TrailerPowerPort;
    }

    part def HitchSystem {
        attribute systemName : String;
        attribute connectionStatus : ConnectionState;

        part vehicle : Vehicle;
        part trailer : Trailer;

        connect vehicle.powerOut to trailer.powerIn;
    }
}