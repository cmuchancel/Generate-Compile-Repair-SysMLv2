package VehicleSystem {
    private import ScalarValues::*;

    enum def MassUnit {
        enum kilogram;
    }

    part def Mass {
        attribute value : Real;
        attribute unit : MassUnit = MassUnit::kilogram;
    }

    part def Vehicle {
        attribute emptyMass : Mass;
        attribute fullMass : Mass;
        part engine : Engine;
        part transmission : Transmission;
        part clutch : Clutch;
    }

    part def Engine {
        attribute torque : Real;
        port torqueOutput : TorquePort;
    }

    part def Transmission {
        port torqueInput : TorquePort;
    }

    part def Clutch {
        port engineSide : TorquePort;
        port transmissionSide : TorquePort;
    }

    port def TorquePort;

    requirement def MassRequirement {
        attribute maxEmptyMass : Mass;
        attribute maxFullMass : Mass;
    }

    requirement def TorqueTransmissionRequirement {
        attribute torqueVariation : String = "Torque varies with speed per performance table";
    }

    part vehicle : Vehicle {
        attribute emptyMass : Mass {
            value = 1500;
        };
        attribute fullMass : Mass {
            value = 2000;
        };
        part engine : Engine;
        part transmission : Transmission;
        part clutch : Clutch;
    }

    connect vehicle.engine.torqueOutput to vehicle.clutch.engineSide;
    connect vehicle.clutch.transmissionSide to vehicle.transmission.torqueInput;

    requirement massReq : MassRequirement {
        attribute maxEmptyMass : Mass {
            value = 1500;
        };
        attribute maxFullMass : Mass {
            value = 2000;
        };
    }

    requirement torqueReq : TorqueTransmissionRequirement;
}