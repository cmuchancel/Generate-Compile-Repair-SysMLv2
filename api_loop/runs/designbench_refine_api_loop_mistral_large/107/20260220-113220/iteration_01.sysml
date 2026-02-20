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
        part frame : VehicleFrame;
        part hitchBall : HitchBall;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        part frame : TrailerFrame;
        part coupler : TrailerCoupler;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    port def PowerPort {
        attribute isConnected : Boolean;
    }

    port vehiclePowerPort : PowerPort on vehicle;
    port trailerPowerPort : PowerPort on trailer;

    item def ElectricalPower;

    connect vehicle.hitchBall to trailer.coupler;
    connect vehiclePowerPort to trailerPowerPort;
}