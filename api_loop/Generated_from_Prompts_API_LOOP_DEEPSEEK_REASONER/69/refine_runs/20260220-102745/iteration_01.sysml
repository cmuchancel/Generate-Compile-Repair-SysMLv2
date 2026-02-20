package VehicleSystem {
    private import ScalarValues::*;

    enum def FuelState {
        full,
        empty
    }

    part def Engine {
        port output torqueOutput : MechanicalPort;
        attribute maxTorque : Real;
        attribute currentSpeed : Real;
    }

    part def Clutch {
        port input torqueInput : MechanicalPort;
        port output torqueOutput : MechanicalPort;
    }

    part def Transmission {
        port input torqueInput : MechanicalPort;
    }

    part def FuelTank {
        attribute capacity : Real = 50.0;
        attribute currentFuelMass : Real = 0.0;
    }

    port def MechanicalPort;

    part def Vehicle {
        attribute dryMass : Real = 1400.0;
        attribute totalMass : Real;
        attribute fuelState : FuelState = FuelState::empty;

        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;
        part fuelTank : FuelTank;

        port powertrainInput : MechanicalPort;
        port powertrainOutput : MechanicalPort;

        connect engine.torqueOutput to clutch.torqueInput;
        connect clutch.torqueOutput to transmission.torqueInput;
    }

    item def MassRequirement {
        attribute maxMassFull : Real = 2000.0;
        attribute maxMassEmpty : Real = 1500.0;
        attribute currentVehicleMass : Real;
    }
}