package PowertrainConfiguration {
  private import ScalarValues::*;

  enum def EngineType {
    enum FourCylinder;
    enum SixCylinder;
  }

  enum def TransmissionType {
    enum Manual;
    enum Automatic;
  }

  part def Engine {
    attribute engineType : EngineType;
  }

  part def FourCylinderEngine {
    attribute engineType : EngineType = EngineType::FourCylinder;
  }

  part def SixCylinderEngine {
    attribute engineType : EngineType = EngineType::SixCylinder;
  }

  part def Transmission {
    attribute transmissionType : TransmissionType;
  }

  part def ManualTransmission {
    attribute transmissionType : TransmissionType = TransmissionType::Manual;
  }

  part def AutomaticTransmission {
    attribute transmissionType : TransmissionType = TransmissionType::Automatic;
  }

  part def Vehicle {
    attribute highPerformance : Boolean;
    part engine : Engine;
    part transmission : Transmission;
  }

  part def FourCylinderManualVehicle {
    attribute highPerformance : Boolean = false;
    part engine : FourCylinderEngine;
    part transmission : ManualTransmission;
  }

  part def SixCylinderAutomaticVehicle {
    attribute highPerformance : Boolean = true;
    part engine : SixCylinderEngine;
    part transmission : AutomaticTransmission;
  }

  part standardVehicle : FourCylinderManualVehicle;
  part highPerformanceVehicle : SixCylinderAutomaticVehicle;
}