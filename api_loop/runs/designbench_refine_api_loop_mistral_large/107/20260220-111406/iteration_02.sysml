package VehicleTrailerCouplingSystem {
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
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    port def PowerPort {
        attribute isConnected : Boolean = false;
    }

    port vehiclePowerPort : PowerPort;
    port trailerPowerPort : PowerPort;

    connect vehicle.hitchBall to trailer.coupler;
    connect vehiclePowerPort to trailerPowerPort;
}