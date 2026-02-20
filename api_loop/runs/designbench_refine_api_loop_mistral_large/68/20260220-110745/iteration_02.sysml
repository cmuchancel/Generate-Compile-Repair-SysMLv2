package VehicleMassAndPower {

    private import ScalarValues::*;

    part def Vehicle {
        attribute curbWeight : Real;
        attribute fuelMass : Real;
        attribute maxMassLimit : Real;
        attribute totalMass : Real = curbWeight + fuelMass;

        requirement MassLimit {
            attribute description : String = "Total mass must not exceed maximum mass limit.";
            attribute satisfied : Boolean = totalMass <= maxMassLimit;
        }

        requirement PositiveFuelMass {
            attribute description : String = "Fuel mass must be a positive value.";
            attribute satisfied : Boolean = fuelMass > 0.0;
        }
    }

    part def Engine {
        attribute speed : Real;
        attribute torque : Real;

        port def TorqueOutput {
            attribute torque : Real;
        }

        port torqueOutput : TorqueOutput;
    }

    part def Clutch {
        port def TorqueInput {
            attribute torque : Real;
        }

        port def TorqueOutput {
            attribute torque : Real;
        }

        port torqueInput : TorqueInput;
        port torqueOutput : TorqueOutput;
    }

    part def Transmission {
        port def TorqueInput {
            attribute torque : Real;
        }

        port torqueInput : TorqueInput;
    }

    part vehicle : Vehicle;
    part engine : Engine;
    part clutch : Clutch;
    part transmission : Transmission;

    connect engine.torqueOutput to clutch.torqueInput;
    connect clutch.torqueOutput to transmission.torqueInput;

    requirement EngineTorque {
        attribute description : String = "Engine must generate torque according to speed curve.";
        attribute satisfied : Boolean = if engine.speed > 0.0 then engine.torque > 0.0 else true;
    }

    requirement TorqueTransmission {
        attribute description : String = "Torque must be effectively transmitted through clutch.";
        attribute satisfied : Boolean = clutch.torqueInput.torque == clutch.torqueOutput.torque;
    }
}