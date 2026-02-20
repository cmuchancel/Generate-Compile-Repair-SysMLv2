package VehicleTrailerConnectionSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum connected;
        enum disconnected;
    }

    part def Vehicle {
        attribute chassis : Chassis;
        attribute towBall : TowBall;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute frame : TrailerFrame;
        attribute coupler : Coupler;
    }

    part def Chassis;
    part def TowBall;
    part def TrailerFrame;
    part def Coupler;

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.towBall to trailer.coupler;
}