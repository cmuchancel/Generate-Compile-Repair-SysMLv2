package VehicleTrailerSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum connected;
        enum disconnected;
    }

    part def Chassis {
        port towBall : TowBallPort;
    }

    part def TrailerFrame {
        port coupler : CouplerPort;
    }

    part def Vehicle {
        part chassis : Chassis;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        attribute frame : TrailerFrame;
    }

    port def TowBallPort;
    port def CouplerPort;

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.chassis.towBall to trailer.frame.coupler;
}