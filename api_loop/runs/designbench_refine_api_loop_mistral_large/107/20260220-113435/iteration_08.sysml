package VehicleTrailerCouplingSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum connected;
        enum disconnected;
    }

    part def HitchBall {
        attribute diameter : Real;
    }

    part def TrailerCoupler {
        attribute maxLoad : Real;
    }

    part def VehicleFrame {
        part hitchBall : HitchBall;
    }

    part def TrailerFrame {
        part trailerCoupler : TrailerCoupler;
    }

    part def Vehicle {
        attribute frame : VehicleFrame;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute frame : TrailerFrame;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.frame.hitchBall to trailer.frame.trailerCoupler;
}