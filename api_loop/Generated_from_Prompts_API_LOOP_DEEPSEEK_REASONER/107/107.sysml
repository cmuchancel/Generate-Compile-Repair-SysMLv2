package VehicleTrailerHitchingSystem {
    private import ScalarValues::*;

    part def VehicleFrame;

    part def TrailerFrame;

    part def HitchBall {
        attribute diameter : Real;
        attribute isEngaged : Boolean = false;
    }

    part def TrailerCoupler {
        attribute openingDiameter : Real;
        attribute isSecured : Boolean = false;
    }

    part def Vehicle {
        part vehicleFrame : VehicleFrame;
        part hitchBall : HitchBall;
    }

    part def Trailer {
        part trailerFrame : TrailerFrame;
        part coupler : TrailerCoupler;
    }

    part def HitchingSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        connect vehicle.hitchBall to trailer.coupler;
    }
}