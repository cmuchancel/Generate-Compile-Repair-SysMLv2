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
        part frame : VehicleFrame;
        part hitchBall : HitchBall;
    }

    part def Trailer {
        part frame : TrailerFrame;
        part coupler : TrailerCoupler;
    }

    part def HitchingSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        connect vehicle.hitchBall to trailer.coupler;
    }
}