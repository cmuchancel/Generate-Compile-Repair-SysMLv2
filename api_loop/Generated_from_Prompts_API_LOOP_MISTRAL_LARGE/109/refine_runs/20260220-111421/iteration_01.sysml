package TowingSystem {
    private import ScalarValues::*;

    enum def HitchState {
        enum ENGAGED;
        enum DISENGAGED;
    }

    part def TowBallDevice {
        attribute diameter : Real;
    }

    part def CouplerMechanism {
        attribute latchState : HitchState = HitchState::DISENGAGED;
    }

    part def VehicleChassis {
        part towBall : TowBallDevice;
    }

    part def TrailerChassis {
        part coupler : CouplerMechanism;
    }

    part vehicle : VehicleChassis;
    part trailer : TrailerChassis;

    connect vehicle.towBall to trailer.coupler;
}