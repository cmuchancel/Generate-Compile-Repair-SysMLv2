package VehicleMassAndDynamics {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    };

    part def Engine {
        attribute mass : Real;
    };

    part def Transmission {
        attribute mass : Real;
    };

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        attribute totalMass : Real;
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute appliedForce : Real;
        attribute time : Real;

        constraint maxMass {
            chassis.mass + engine.mass + transmission.mass <= 2500.0
        };

        constraint positiveMass {
            chassis.mass > 0.0 and engine.mass > 0.0 and transmission.mass > 0.0
        };

        constraint dynamicConstraint {
            appliedForce * time == (finalVelocity - initialVelocity) * (chassis.mass + engine.mass + transmission.mass)
        };
    };
}