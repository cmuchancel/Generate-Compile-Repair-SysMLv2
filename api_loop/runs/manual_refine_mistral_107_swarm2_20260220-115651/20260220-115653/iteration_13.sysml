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

    port def PowerPort;

    part def Vehicle {
        part frame : VehicleFrame;
        part hitchBall : HitchBall;
        port powerPort : PowerPort;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        part frame : TrailerFrame;
        part coupler : TrailerCoupler;
        port powerPort : PowerPort;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.hitchBall to trailer.coupler;
    connect vehicle.powerPort to trailer.powerPort;
}