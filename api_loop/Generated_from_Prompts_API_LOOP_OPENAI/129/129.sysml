package VehicleSafetyAntiTheftCatalogue {

  private import ScalarValues::*;

  part def FrontSeatBelts {
    attribute affectsSafety : Boolean = true;
    attribute affectsAntiTheft : Boolean = false;
    attribute mandatorySafety : Boolean = true;
  }

  part def Bumpers {
    attribute affectsSafety : Boolean = true;
    attribute affectsAntiTheft : Boolean = false;
    attribute mandatorySafety : Boolean = true;
  }

  part def DriverAirbag {
    attribute affectsSafety : Boolean = true;
    attribute affectsAntiTheft : Boolean = false;
    attribute mandatorySafety : Boolean = false;
  }

  part def AntiLockBrakingSystem {
    attribute affectsSafety : Boolean = true;
    attribute affectsAntiTheft : Boolean = false;
    attribute mandatorySafety : Boolean = false;
  }

  part def AlarmSystem {
    attribute affectsSafety : Boolean = false;
    attribute affectsAntiTheft : Boolean = true;
    attribute mandatorySafety : Boolean = false;
  }

  part def KeylessEntrySystem {
    attribute affectsSafety : Boolean = false;
    attribute affectsAntiTheft : Boolean = true;
    attribute mandatorySafety : Boolean = false;
  }

  part def SafetyAntiTheftControlModule {
    attribute affectsSafety : Boolean = true;
    attribute affectsAntiTheft : Boolean = true;
    attribute mandatorySafety : Boolean = false;
  }

  part def VehicleSafetyAntiTheftCatalogue {
    part frontSeatBelts : FrontSeatBelts;
    part bumpers : Bumpers;
    part driverAirbag : DriverAirbag;
    part antiLockBrakingSystem : AntiLockBrakingSystem;
    part alarmSystem : AlarmSystem;
    part keylessEntrySystem : KeylessEntrySystem;
    part safetyAntiTheftControlModule : SafetyAntiTheftControlModule;
  }

  part def Vehicle {
    part frontSeatBelts : FrontSeatBelts;
    part bumpers : Bumpers;
    part driverAirbag : DriverAirbag;
    part antiLockBrakingSystem : AntiLockBrakingSystem;
    part alarmSystem : AlarmSystem;
    part keylessEntrySystem : KeylessEntrySystem;
    part safetyAntiTheftControlModule : SafetyAntiTheftControlModule;
    part catalogue : VehicleSafetyAntiTheftCatalogue;
  }

}