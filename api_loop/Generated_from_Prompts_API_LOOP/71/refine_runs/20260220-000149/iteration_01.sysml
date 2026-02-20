package VehicleSystem {
    private import ScalarValues::*;

    item def TorqueItem {
        attribute magnitude : Real;
        attribute speed : Real;
    }

    port def TorquePort {
        item torque : TorqueItem;
    }

    part def Engine {
        attribute maxTorque : Real;
        attribute speed1 : Real;
        attribute torqueAtSpeed1 : Real;
        attribute speed2 : Real;
        attribute torqueAtSpeed2 : Real;
        port outTorque : TorquePort;
    }

    part def Clutch {
        port inTorque : TorquePort;
        port outTorque : TorquePort;
    }

    part def Gearbox {
        port inTorque : TorquePort;
    }

    part def VehicleMassCondition {
        attribute curbMass : Real;
        attribute fuelMass : Real;
        attribute totalMass : Real;
    }

    part def Vehicle {
        part engine : Engine {
            attribute maxTorque = 350;
            attribute speed1 = 2000;
            attribute torqueAtSpeed1 = 300;
            attribute speed2 = 4000;
            attribute torqueAtSpeed2 = 280;
        }
        part clutch : Clutch;
        part gearbox : Gearbox;

        part fullFuel : VehicleMassCondition {
            attribute curbMass = 1400;
            attribute fuelMass = 500;
            attribute totalMass = 1900;
        }

        part emptyFuel : VehicleMassCondition {
            attribute curbMass = 1400;
            attribute fuelMass = 0;
            attribute totalMass = 1400;
        }

        connect engine.outTorque to clutch.inTorque;
        connect clutch.outTorque to gearbox.inTorque;
    }
}