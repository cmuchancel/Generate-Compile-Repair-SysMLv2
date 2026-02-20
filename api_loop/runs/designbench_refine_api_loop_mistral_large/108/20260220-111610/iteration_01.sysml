package VehicleTrailerSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        connected
        disconnected
    }

    part def Vehicle {
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
        part chassis : Chassis;
    }

    part def Chassis {
        port towBall : TowBallPort;
    }

    part def Trailer {
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
        part frame : TrailerFrame;
    }

    part def TrailerFrame {
        port coupler : CouplerPort;
    }

    port def TowBallPort;
    port def CouplerPort;

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.chassis.towBall to trailer.frame.coupler;
}