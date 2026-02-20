package VehicleSystem {
    private import ScalarValues::*;

    port def TorquePort {
        attribute torque : Real;
        attribute speed : Real;
    }

    part def Engine {
        attribute maxTorque : Real = 350;
        attribute speed1 : Real = 2000;
        attribute torqueAtSpeed1 : Real = 300;
        attribute speed2 : Real = 4000;
        attribute torqueAtSpeed2 : Real = 280;
        port outTorque : TorquePort;
    }

    part def Clutch {
        port inTorque : TorquePort;
        port outTorque : TorquePort;
    }

    part def Gearbox {
        port inTorque : TorquePort;
    }

    part def FullFuelMass {
        attribute curbMass : Real = 1400;
        attribute fuelMass : Real = 500;
        attribute totalMass : Real = 1900;
    }

    part def EmptyFuelMass {
        attribute curbMass : Real = 1400;
        attribute fuelMass : Real = 0;
        attribute totalMass : Real = 1400;
    }

    part def Vehicle {
        part engine : Engine;
        part clutch : Clutch;
        part gearbox : Gearbox;

        part fullFuel : FullFuelMass;
        part emptyFuel : EmptyFuelMass;

        connect engine.outTorque to clutch.inTorque;
        connect clutch.outTorque to gearbox.inTorque;
    }
}