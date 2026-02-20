package VehicleMassAndDynamics {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        attribute totalMass : Real = chassis.mass + engine.mass + transmission.mass;
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute appliedForce : Real;
        attribute time : Real;

        constraint {
            chassis.mass + engine.mass + transmission.mass <= 2500.0
        }

        constraint {
            chassis.mass > 0.0;
            engine.mass > 0.0;
            transmission.mass > 0.0;
            totalMass > 0.0
        }

        constraint {
            (finalVelocity - initialVelocity) * totalMass == appliedForce * time
        }
    }
}