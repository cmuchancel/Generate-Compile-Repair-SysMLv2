package VehicleSystem {
    private import ScalarValues::*;

    enum def FuelState {
        enum full;
        enum empty;
    }

    port def MechanicalPort;

    part def Engine {
        port torqueOut : MechanicalPort;
        attribute maxTorque : Real;
        attribute currentSpeed : Real;
    }

    part def Clutch {
        port torqueIn : MechanicalPort;
        port torqueOut : MechanicalPort;
    }

    part def Transmission {
        port torqueIn : MechanicalPort;
    }

    part def FuelTank {
        attribute capacity : Real = 50.0;
        attribute currentFuelMass : Real = 0.0;
    }

    part def Vehicle {
        attribute dryMass : Real = 1400.0;
        attribute totalMass : Real;
        attribute fuelState : FuelState = FuelState::empty;

        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;
        part fuelTank : FuelTank;

        connect engine.torqueOut to clutch.torqueIn;
        connect clutch.torqueOut to transmission.torqueIn;
    }

    item def MassRequirement {
        attribute maxMassFull : Real = 2000.0;
        attribute maxMassEmpty : Real = 1500.0;
        attribute currentVehicleMass : Real;
    }
}