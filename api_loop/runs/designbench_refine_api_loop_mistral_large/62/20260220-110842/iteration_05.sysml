package VehicleMassManagement {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Gearbox {
        attribute mass : Real;
    }

    item def OverweightWarning;

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part gearbox : Gearbox;
        attribute totalMass : Real;
        attribute maxPermittedMass : Real = 2500.0;
        attribute isOverweight : Boolean;
        attribute overweightWarning : OverweightWarning;
    }

    part vehicle : Vehicle {
        part chassis : Chassis {
            attribute mass : Real = 0.0;
        }
        part engine : Engine {
            attribute mass : Real = 0.0;
        }
        part gearbox : Gearbox {
            attribute mass : Real = 0.0;
        }
    }

    calc def computeTotalMass {
        attribute total : Real = vehicle.chassis.mass + vehicle.engine.mass + vehicle.gearbox.mass;
        vehicle.totalMass = total;
    }

    calc def checkOverweight {
        attribute overweight : Boolean = vehicle.totalMass > vehicle.maxPermittedMass;
        vehicle.isOverweight = overweight;
        if (overweight) then {
            vehicle.overweightWarning = new OverweightWarning;
        }
    }
}