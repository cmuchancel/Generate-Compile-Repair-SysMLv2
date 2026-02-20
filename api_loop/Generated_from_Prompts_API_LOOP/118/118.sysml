package AutomobileSystem {
  private import ScalarValues::*;

  enum def MassGrade {
    enum kg2000;
    enum kg2500;
  }

  enum def FuelStatus {
    enum full;
    enum empty;
  }

  item def Torque {
    attribute value : Real;
  }

  port def DrivePort {
    attribute torque : Torque;
  }

  part def Engine {
    attribute outputTorque : Real;
    port driveOut : DrivePort;
  }

  part def Transmission {
    attribute inputTorque : Real;
    port driveIn : DrivePort;
  }

  part def Powertrain {
    part engine : Engine;
    part transmission : Transmission;
    connect engine.driveOut to transmission.driveIn;
  }

  part def Vehicle {
    attribute actualMass : Real;
    attribute maxMass : Real;
    attribute massGrade : MassGrade;
    attribute fuelStatus : FuelStatus;
    attribute reliabilityLevel : Real;
    part powertrain : Powertrain;
  }

  part def AutomobileSpecSystem {
    part vehicle2000 : Vehicle {
      actualMass = 1980.0;
      maxMass = 2000.0;
      massGrade = MassGrade::kg2000;
      fuelStatus = FuelStatus::full;
      reliabilityLevel = 0.98;
    }
    part vehicle2500 : Vehicle {
      actualMass = 2450.0;
      maxMass = 2500.0;
      massGrade = MassGrade::kg2500;
      fuelStatus = FuelStatus::empty;
      reliabilityLevel = 0.97;
    }
  }

  part system : AutomobileSpecSystem;
}