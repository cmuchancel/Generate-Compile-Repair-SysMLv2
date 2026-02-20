package VehicleRequirements {
    private import ScalarValues::*;

    enum def OperatingCondition {
        enum FullFuelTank;
        enum EmptyFuelTank;
    }

    part def Vehicle {
        attribute curbMass : Real;
        attribute fuelMass : Real;
        attribute totalMass : Real;
        attribute maxMassFull : Real = 2000.0;
        attribute maxMassEmpty : Real = 1500.0;

        part fuelTank : FuelTank;
        part engine : Engine;
        part clutch : Clutch;
        part gearbox : Gearbox;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentFuelMass : Real;
    }

    part def Engine {
        attribute torque : Real;
        attribute speed : Real;
    }

    part def Clutch {
        port clutchInput : ClutchInputPort;
        port clutchOutput : ClutchOutputPort;
    }

    port def ClutchInputPort;
    port def ClutchOutputPort;

    part def Gearbox;

    requirement VehicleMassLimit {
        attribute condition : OperatingCondition;
        attribute maxAllowedMass : Real;
    }

    requirement PowerTransmission {
        attribute requiredTorque : Real;
    }

    part vehicle : Vehicle;

    connect vehicle.engine to vehicle.clutch.clutchInput;
    connect vehicle.clutch.clutchOutput to vehicle.gearbox;

    attribute fullCondition : OperatingCondition = OperatingCondition::FullFuelTank;
    attribute emptyCondition : OperatingCondition = OperatingCondition::EmptyFuelTank;

    requirement vehicleMassFull : VehicleMassLimit {
        condition = fullCondition;
        maxAllowedMass = 2000.0;
    }

    requirement vehicleMassEmpty : VehicleMassLimit {
        condition = emptyCondition;
        maxAllowedMass = 1500.0;
    }

    requirement torqueTransmission : PowerTransmission;
}