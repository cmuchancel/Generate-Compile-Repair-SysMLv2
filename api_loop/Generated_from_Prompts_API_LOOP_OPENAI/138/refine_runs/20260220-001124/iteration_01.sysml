package PowertrainConfiguration
  private import ScalarValues::*;

  enum def EngineType {
    FourCylinder;
    SixCylinder;
  }

  enum def TransmissionType {
    Manual;
    Automatic;
  }

  part def Engine {
    attribute type : EngineType;
  }

  part def FourCylinderEngine : Engine {
    attribute type : EngineType = FourCylinder;
  }

  part def SixCylinderEngine : Engine {
    attribute type : EngineType = SixCylinder;
  }

  part def Transmission {
    attribute type : TransmissionType;
  }

  part def ManualTransmission : Transmission {
    attribute type : TransmissionType = Manual;
  }

  part def AutomaticTransmission : Transmission {
    attribute type : TransmissionType = Automatic;
  }

  part def Vehicle {
    attribute highPerformance : Boolean;
    part engine : Engine;
    part transmission : Transmission;
  }

  part def FourCylinderManualVehicle : Vehicle {
    attribute highPerformance : Boolean = false;
    part engine : FourCylinderEngine;
    part transmission : ManualTransmission;
  }

  part def SixCylinderAutomaticVehicle : Vehicle {
    attribute highPerformance : Boolean = true;
    part engine : SixCylinderEngine;
    part transmission : AutomaticTransmission;
  }

  part standardVehicle : FourCylinderManualVehicle;
  part highPerformanceVehicle : SixCylinderAutomaticVehicle;
end package