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
        attribute hitchBall : HitchBall;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
        port powerPort : PowerPort;
    }

    part def Trailer {
        part frame : TrailerFrame;
        attribute coupler : TrailerCoupler;
        attribute connectionStatus : ConnectionStatus = ConnectionStatus::disconnected;
        port powerPort : PowerPort;
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

    part couplingSystem : CouplingSystem {
        connect vehicle.powerPort to trailer.powerPort;
    }
}