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

    part def Vehicle {
        attribute vehicleFrame : VehicleFrame;
        attribute hitchBall : HitchBall;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute trailerFrame : TrailerFrame;
        attribute coupler : TrailerCoupler;
    }

    part vehiclePart : Vehicle;
    part trailerPart : Trailer;

    port def PowerPort {
        attribute isConnected : Boolean;
    }

    port vehiclePort : PowerPort;
    port trailerPort : PowerPort;

    connect vehiclePart.hitchBall to trailerPart.coupler;
    connect vehiclePort to trailerPort;
}