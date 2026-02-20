package VehicleTrailerCoupling {

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
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute frame : TrailerFrame;
        attribute coupler : TrailerCoupler;
    }

    port def PowerPort;

    part vehicle : Vehicle {
        port powerPort : PowerPort;
    }

    part trailer : Trailer {
        port powerPort : PowerPort;
    }

    connect vehicle.hitchBall to trailer.coupler;
    connect vehicle.powerPort to trailer.powerPort;

}