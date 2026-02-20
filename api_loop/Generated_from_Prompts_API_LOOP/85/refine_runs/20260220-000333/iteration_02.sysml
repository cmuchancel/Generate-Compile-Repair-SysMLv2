package VehicleFeatureAnnotationSystem {
  private import ScalarValues::*;

  enum def FeatureFunction {
    safety;
    security;
    protection;
  }

  part def SeatBelts {
    attribute functionType : FeatureFunction = FeatureFunction::safety;
  }

  part def FrontSeats {
    attribute functionType : FeatureFunction = FeatureFunction::safety;
  }

  part def DriverAirbag {
    attribute functionType : FeatureFunction = FeatureFunction::safety;
  }

  part def AlarmSystem {
    attribute functionType : FeatureFunction = FeatureFunction::security;
  }

  part def BodyShell {
    attribute functionType : FeatureFunction = FeatureFunction::protection;
  }

  part def Bumpers {
    attribute functionType : FeatureFunction = FeatureFunction::safety;
  }

  part def KeylessEntryDevice {
    attribute functionType : FeatureFunction = FeatureFunction::security;
  }

  part def InteriorAssembly {
    part seatBelts : SeatBelts;
    part frontSeats : FrontSeats;
    part driverAirbag : DriverAirbag;
    part alarmSystem : AlarmSystem;
  }

  part def BodyAssembly {
    part bodyShell : BodyShell;
    part bumpers : Bumpers;
    part keylessEntryDevice : KeylessEntryDevice;
  }

  part def Vehicle {
    part interior : InteriorAssembly;
    part bodyAssembly : BodyAssembly;
  }

  part vehicle : Vehicle;
}