package TrailerHitchSystem {
    private import ScalarValues::*;

    enum def ConnectionState {
        enum Connected;
        enum Disconnected;
    }

    port def HitchPort;

    part def TowBall {
        port hitch : HitchPort;
    }

    part def Coupler {
        port hitch : HitchPort;
        attribute canEngage : Boolean = true;
        attribute canDisengage : Boolean = true;
        attribute engaged : Boolean = false;
    }

    part def Chassis {
    }

    part def TrailerChassis {
    }

    part def Vehicle {
        part chassis : Chassis;
        part towBall : TowBall;
    }

    part def Trailer {
        part trailerChassis : TrailerChassis;
        part coupler : Coupler;
    }

    part def VehicleTrailerSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        attribute connectionState : ConnectionState = ConnectionState::Disconnected;

        connect vehicle.towBall.hitch to trailer.coupler.hitch;
    }

    part system : VehicleTrailerSystem;
}