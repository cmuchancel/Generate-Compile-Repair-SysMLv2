package VehicleTrailerConnectionSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum connected;
        enum disconnected;
    }

    part def Chassis;
    part def TowBall;
    part def TrailerFrame;
    part def Coupler;

    part def Vehicle {
        part chassis : Chassis;
        part towBall : TowBall;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        part frame : TrailerFrame;
        part coupler : Coupler;
    }

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.towBall to trailer.coupler;
}