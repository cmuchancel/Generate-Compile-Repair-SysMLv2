package VehicleTrailerCouplingSystem {
    private import ScalarValues::*;

    enum def CouplingType {
        enum TowHitch;
        enum BallConnector;
    }

    part def Hitch {
        attribute couplingType : CouplingType;
        port couplingBall : CouplingBallPort;
    }

    part def Vehicle {
        port couplingInterface : CouplingInterfacePort;
    }

    part def Trailer {
        port couplingConnector : CouplingConnectorPort;
    }

    port def CouplingBallPort;
    port def CouplingInterfacePort;
    port def CouplingConnectorPort;

    part couplingSystem {
        part vehicle : Vehicle;
        part hitch : Hitch;
        part trailer : Trailer;

        connect vehicle.couplingInterface to hitch.couplingBall;
        connect hitch.couplingBall to trailer.couplingConnector;
    }
}