package VehicleSafetySecurity {
  private import ScalarValues::*;

  enum def SafetyClass {
    enum SafetyCritical;
    enum SecurityCritical;
    enum Optional;
  }

  part def SeatBelt {
    attribute isMandatory : Boolean = true;
    attribute safetyClass : SafetyClass = SafetyCritical;
  }

  part def Airbag {
    attribute isMandatory : Boolean = false;
    attribute safetyClass : SafetyClass = SafetyCritical;
  }

  part def Alarm {
    attribute isMandatory : Boolean = true;
    attribute safetyClass : SafetyClass = SecurityCritical;
  }

  part def Bumper {
    attribute isMandatory : Boolean = true;
    attribute safetyClass : SafetyClass = SafetyCritical;
  }

  part def KeylessEntry {
    attribute isMandatory : Boolean = true;
    attribute safetyClass : SafetyClass = SecurityCritical;
  }

  part def Wheel {
    attribute isMandatory : Boolean = true;
    attribute safetyClass : SafetyClass = SafetyCritical;
  }

  part def ABS {
    attribute isMandatory : Boolean = false;
    attribute safetyClass : SafetyClass = SafetyCritical;
  }

  part def Interior {
    part seatBelts : SeatBelt[2];
    part airbag : Airbag;
    part alarm : Alarm;
  }

  part def Body {
    part bumper : Bumper;
    part keylessEntry : KeylessEntry;
  }

  part def Chassis {
    part wheels : Wheel[2];
    part absSets : ABS[2];
  }

  part def SafetySecurityManager {
    attribute classificationEnabled : Boolean = true;
  }

  part def Vehicle {
    part interior : Interior;
    part body : Body;
    part chassis : Chassis;
    part manager : SafetySecurityManager;
  }

  part vehicle : Vehicle;
}