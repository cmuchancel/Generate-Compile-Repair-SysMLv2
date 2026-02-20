package VehicleTrailerHitchSystem {
    private import ScalarValues::*;

    enum def ConnectionState {
        enum Connected;
        enum Disconnected;
    }

    port def TowBallPort {
        attribute connectionState : ConnectionState;
    }

    port def CouplerPort {
        attribute connectionState : ConnectionState;
    }

    part def TowBallDevice {
        attribute ballDiameter : Real;
        attribute maxTowLoad : Real;
        port towBallInterface : TowBallPort;
    }

    part def Chassis {
        attribute chassisLength : Real;
        attribute chassisWidth : Real;
        part towBall : TowBallDevice;
    }

    part def CouplerMechanism {
        attribute isLatched : Boolean;
        attribute couplerDiameter : Real;
        port couplerInterface : CouplerPort;
    }

    part def TrailerChassis {
        attribute trailerLength : Real;
        attribute trailerWidth : Real;
        part coupler : CouplerMechanism;
    }

    part def Vehicle {
        part vehicleChassis : Chassis;
    }

    part def Trailer {
        part trailerBody : TrailerChassis;
    }

    part def HitchSystem {
        attribute connectionStatus : ConnectionState;
        attribute isHitched : Boolean;

        part vehicle : Vehicle;
        part trailer : Trailer;

        connect vehicle.vehicleChassis.towBall.towBallInterface
            to trailer.trailerBody.coupler.couplerInterface;
    }
}