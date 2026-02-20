package VehicleTrailerHitchSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum connected;
        enum disconnected;
    }

    part def VehicleFrame;
    part def TrailerFrame;
    part def HitchBall;
    part def TrailerCoupler;

    port def PowerPort {
        attribute voltage : Real;
        attribute current : Real;
    }

    part def Vehicle {
        attribute frame : VehicleFrame;
        part hitchBall : HitchBall;
        port powerOut : PowerPort;
        attribute status : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute frame : TrailerFrame;
        part coupler : TrailerCoupler;
        port powerIn : PowerPort;
        attribute status : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.hitchBall to trailer.coupler;
    connect vehicle.powerOut to trailer.powerIn;
}