package VehicleWeightManagement {
    private import ScalarValues::*;

    part def Chassis {
        attribute weight : Real;
    }

    part def Engine {
        attribute weight : Real;
    }

    part def Transmission {
        attribute weight : Real;
    }

    part def WeightMonitor {
        attribute maxAllowedWeight : Real = 2500.0;
        attribute totalWeight : Real = 0.0;
        attribute warningActive : Boolean = false;

        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;

        attribute calculateTotalWeight() : Real {
            totalWeight = chassis.weight + engine.weight + transmission.weight;
            return totalWeight;
        }

        attribute checkWeightLimit() : Boolean {
            calculateTotalWeight();
            if totalWeight > maxAllowedWeight then {
                warningActive = true;
            } else {
                warningActive = false;
            }
            return warningActive;
        }
    }

    part vehicleWeightSystem : WeightMonitor;
}