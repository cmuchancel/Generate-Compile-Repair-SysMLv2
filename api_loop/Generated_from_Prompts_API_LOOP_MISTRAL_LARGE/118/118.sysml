package AutomobileRequirements {
    private import ScalarValues::*;

    enum def MassGrade {
        enum GRADE_2000KG;
        enum GRADE_2500KG;
    }

    enum def FuelStatus {
        enum FULL;
        enum EMPTY;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute maxMass : Real;
        attribute massGrade : MassGrade;
        attribute fuelStatus : FuelStatus;
        attribute isOverloaded : Boolean = false;

        port driveInterface : DrivePort;

        part engine : Engine;
        part transmission : Transmission;
    }

    part def Engine {
        attribute torqueOutput : Real;
        port powerOutput : PowerPort;
    }

    part def Transmission {
        port powerInput : PowerPort;
    }

    port def DrivePort;
    port def PowerPort;

    item def Torque;

    part vehicle2000kg : Vehicle {
        mass = 2000.0;
        maxMass = 2000.0;
        massGrade = MassGrade::GRADE_2000KG;
        fuelStatus = FuelStatus::FULL;
    }

    part vehicle2500kg : Vehicle {
        mass = 2500.0;
        maxMass = 2500.0;
        massGrade = MassGrade::GRADE_2500KG;
        fuelStatus = FuelStatus::EMPTY;
    }

    connect vehicle2000kg.engine.powerOutput to vehicle2000kg.transmission.powerInput;
    connect vehicle2000kg.driveInterface to vehicle2000kg.transmission;

    connect vehicle2500kg.engine.powerOutput to vehicle2500kg.transmission.powerInput;
    connect vehicle2500kg.driveInterface to vehicle2500kg.transmission;
}