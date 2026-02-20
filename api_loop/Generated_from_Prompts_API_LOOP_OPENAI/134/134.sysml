package BloodGlucoseMeterSystem {
  private import ScalarValues::*;

  enum def BatteryStatus { 
    OK;
    LOW;
    DEPLETED;
  }

  enum def MeasurementStatus { 
    NORMAL;
    ABNORMAL;
  }

  port def StatusPort {
    attribute status : BatteryStatus;
  }

  port def MeasurementPort {
    attribute status : MeasurementStatus;
  }

  port def AlarmPort {
    attribute alarmActive : Boolean;
  }

  part def BatteryMonitor {
    port statusOut : StatusPort;
    attribute autoDetect : Boolean = true;
  }

  part def AlarmSystem {
    port statusIn : StatusPort;
    port alarmOut : AlarmPort;
    attribute alertUser : Boolean = true;
  }

  part def GlucoseMeasurement {
    port measurementOut : MeasurementPort;
    attribute measurementFunction : Boolean = true;
  }

  part def EmergencyResponse {
    port measurementIn : MeasurementPort;
    port alarmIn : AlarmPort;
    attribute responseReady : Boolean = true;
  }

  part def Management {
    attribute approvedRequirements : Boolean = true;
    attribute keyRiskManaged : Boolean = true;
  }

  part def System {
    part batteryMonitor : BatteryMonitor;
    part alarmSystem : AlarmSystem;
    part glucoseMeasurement : GlucoseMeasurement;
    part emergencyResponse : EmergencyResponse;
    part management : Management;

    connect batteryMonitor.statusOut to alarmSystem.statusIn;
    connect glucoseMeasurement.measurementOut to emergencyResponse.measurementIn;
    connect alarmSystem.alarmOut to emergencyResponse.alarmIn;
  }
}