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
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
        part chassis : Chassis;
    }

    part def Trailer {
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
        part trailerFrame : TrailerFrame;
    }

    port def TowBallPort;
    port def CouplerPort;

    part vehicle : Vehicle;
    part trailer : Trailer;

    connect vehicle.chassis.towBall to trailer.trailerFrame.coupler;
}