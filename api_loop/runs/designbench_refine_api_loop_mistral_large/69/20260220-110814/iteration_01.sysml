package VehicleRequirements {
    private import ScalarValues::*;

    enum def MassCondition {
        enum fullFuelTank;
        enum emptyFuelTank;
    }

    part def Vehicle {
        attribute dryMass : Real;
        attribute fuelMass : Real;
        attribute totalMass : Real;
        attribute maxFullMass : Real = 2000.0;
        attribute maxEmptyMass : Real = 1500.0;

        part engine : Engine;
        part transmission : Transmission;
        part clutch : Clutch;
        part fuelTank : FuelTank;
    }

    part def Engine {
        attribute torque : Real;
        port clutchInterface : ClutchPort;
    }

    part def Transmission {
        port clutchInterface : ClutchPort;
    }

    part def Clutch {
        port engineSide : ClutchPort;
        port transmissionSide : ClutchPort;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentFuelMass : Real;
    }

    port def ClutchPort;

    item def Torque;

    part vehicle : Vehicle;

    connect vehicle.engine.clutchInterface to vehicle.clutch.engineSide;
    connect vehicle.clutch.transmissionSide to vehicle.transmission.clutchInterface;
}