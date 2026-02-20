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
        part frame : VehicleFrame;
        part hitchBall : HitchBall;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def Trailer {
        part frame : TrailerFrame;
        part coupler : TrailerCoupler;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
    }

    part def CouplingSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;

        connect vehicle.hitchBall to trailer.coupler;
    }

    item def PowerSignal;

    port def PowerPort {
        in powerIn : PowerSignal;
        out powerOut : PowerSignal;
    }

    part vehicle : Vehicle {
        port powerPort : PowerPort;
    }

    part trailer : Trailer {
        port powerPort : PowerPort;
    }

    part couplingSystem : CouplingSystem {
        connect vehicle.powerPort to trailer.powerPort;
    }
}