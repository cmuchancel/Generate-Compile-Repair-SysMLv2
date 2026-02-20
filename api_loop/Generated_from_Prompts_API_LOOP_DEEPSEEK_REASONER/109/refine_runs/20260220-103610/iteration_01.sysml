package TowingSystem {
    private import ScalarValues::*;

    enum def ConnectionState {
        enum ENGAGED;
        enum DISENGAGED;
    }

    part def Chassis;

    part def TowBall {
        attribute state : ConnectionState = ConnectionState::DISENGAGED;
    }

    part def TrailerChassis;

    part def Coupler {
        attribute state : ConnectionState = ConnectionState::DISENGAGED;
    }

    part def Vehicle {
        part chassis : Chassis;
        part towBall : TowBall;
    }

    part def Trailer {
        part trailerChassis : TrailerChassis;
        part coupler : Coupler;
    }

    port def MechanicalAttachment;

    part def TowingSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        port vehicleHitch : MechanicalAttachment;
        port trailerHitch : MechanicalAttachment;

        connect vehicleHitch to trailerHitch;
    }
}