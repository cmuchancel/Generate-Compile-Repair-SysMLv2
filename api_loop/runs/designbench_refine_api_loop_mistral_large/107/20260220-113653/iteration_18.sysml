package VehicleTrailerHitching {

    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum connected;
        enum disconnected;
    }

    part def VehicleFrame;
    part def TrailerFrame;
    part def HitchBall;
    part def TrailerCoupler;

    part def Vehicle {
        attribute frame : VehicleFrame;
        attribute hitchBall : HitchBall;
        attribute status : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute frame : TrailerFrame;
        attribute coupler : TrailerCoupler;
        attribute status : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.hitchBall to trailer.coupler;

}