package VehicleSystem {
    private import ScalarValues::*;

    attribute def MassValue;
    attribute def TorqueValue;
    attribute def SpeedValue;

    enum def FuelState {
        enum Full;
        enum Empty;
    }

    part def FuelTank {
        attribute fuelMass : Real;
        attribute fuelState : FuelState;
    }

    part def Engine {
        attribute generatedTorque : Real;
        attribute operatingSpeed : Real;
    }

    part def Clutch {
        attribute transmittedTorque : Real;
    }

    part def Gearbox {
        attribute receivedTorque : Real;
    }

    part def Vehicle {
        attribute curbMass : Real;
        attribute totalMassFullTank : Real;
        attribute totalMassEmptyTank : Real;
        attribute massLimitFullTank : Real = 2000.0;
        attribute massLimitEmptyTank : Real = 1500.0;

        part fuelTank : FuelTank;
        part engine : Engine;
        part clutch : Clutch;
        part gearbox : Gearbox;
    }

    port def TorquePort {
        attribute torque : Real;
    }

    part def PowerTrainSystem {
        attribute engineTorque : Real;
        attribute clutchTorque : Real;
        attribute gearboxTorque : Real;

        part trainEngine : Engine;
        part trainClutch : Clutch;
        part trainGearbox : Gearbox;

        port engineOut : TorquePort;
        port clutchIn : TorquePort;
        port clutchOut : TorquePort;
        port gearboxIn : TorquePort;

        connect engineOut to clutchIn;
        connect clutchOut to gearboxIn;
    }

    part def MassConstraintFullTank {
        attribute curbMass : Real;
        attribute fuelMass : Real;
        attribute totalMass : Real;
        attribute massLimit : Real = 2000.0;
    }

    part def MassConstraintEmptyTank {
        attribute curbMass : Real;
        attribute fuelMass : Real = 0.0;
        attribute totalMass : Real;
        attribute massLimit : Real = 1500.0;
    }

    part vehicleInstance : Vehicle {
        attribute :>> curbMass = 1200.0;
        attribute :>> totalMassFullTank = 1800.0;
        attribute :>> totalMassEmptyTank = 1200.0;
    }

    part fullTankCheck : MassConstraintFullTank {
        attribute :>> curbMass = 1200.0;
        attribute :>> fuelMass = 600.0;
        attribute :>> totalMass = 1800.0;
    }

    part emptyTankCheck : MassConstraintEmptyTank {
        attribute :>> curbMass = 1200.0;
        attribute :>> totalMass = 1200.0;
    }

    part powerTrain : PowerTrainSystem {
        attribute :>> engineTorque = 350.0;
        attribute :>> clutchTorque = 350.0;
        attribute :>> gearboxTorque = 350.0;
    }
}