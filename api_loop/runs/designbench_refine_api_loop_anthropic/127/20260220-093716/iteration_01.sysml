package PowertrainTorqueSystem {
    private import ScalarValues::*;

    part def Engine {
        attribute torqueOutput : Real;
    }

    part def TorqueGenerator {
        attribute generatedTorque : Real;
    }

    part def PowertrainSystem {
        part engine : Engine {
            attribute torqueOutput : Real = 250.0;
        }
        part torqueGen : TorqueGenerator {
            attribute generatedTorque : Real = 250.0;
        }
        attribute totalTorque : Real = 250.0;
        attribute torqueGreaterThanZero : Boolean = true;
    }

    part def Vehicle {
        part powertrain : PowertrainSystem;
        attribute powerOutputSatisfied : Boolean = true;
    }

    part vehicle : Vehicle;
}