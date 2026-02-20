package VehicleSafetySecurityModel {
  private import ScalarValues::*;

  part def AlarmSystem {
    attribute securityFeature : Boolean = true;
  }

  part def SeatBelt {
    attribute safetyFeature : Boolean = true;
  }

  part def FrontRowSeat {
  }

  part def DriverAirbag {
    attribute safetyFeature : Boolean = true;
  }

  part def VehicleBody {
  }

  part def Bumper {
    attribute safetyFeature : Boolean = true;
  }

  part def KeylessEntrySystem {
    attribute securityFeature : Boolean = true;
  }

  part def Interior {
    part alarmSystem : AlarmSystem;
    part seatBelts : SeatBelt;
    part frontRowSeats : FrontRowSeat;
    part driverAirbag : DriverAirbag;
  }

  part def BodyAssembly {
    part vehicleBody : VehicleBody;
    part bumper : Bumper;
    part keylessEntrySystem : KeylessEntrySystem;
  }

  part def VehicleSafetySecuritySystem {
    part interior : Interior;
    part bodyAssembly : BodyAssembly;
    attribute safetyIntegrated : Boolean = true;
    attribute securityIntegrated : Boolean = true;
  }

  part vehicleSystem : VehicleSafetySecuritySystem;
}