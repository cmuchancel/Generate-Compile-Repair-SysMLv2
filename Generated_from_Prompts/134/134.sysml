package 'GlucoseMeterSafetySystem' {
	private import ScalarValues::*;
	
	action def DetectBatteryStatus {
		out batteryLevelDescription : String;
		out batteryRechargeable : Boolean;
	}
	
	action def IssueAlarm {
		in batteryLevelDescription : String;
		in batteryRechargeable : Boolean;
	}
	
	action def ExecuteEmergencyResponse {
		in measurementFailure : Boolean;
	}
	
	part def BloodGlucoseMeter {
		attribute measurementOperational : Boolean;
		part batteryMonitor {
			attribute preventiveMeasureEnabled : Boolean = true;
			attribute lastBatteryLevel : String;
			attribute lastRechargeableState : Boolean;
			perform action detectBattery : DetectBatteryStatus {
				out batteryLevelDescription = lastBatteryLevel;
				out batteryRechargeable = lastRechargeableState;
			}
		}
		part userAlarmInterface {
			perform action sendAlarm : IssueAlarm {
				in batteryLevelDescription = batteryMonitor.lastBatteryLevel;
				in batteryRechargeable = batteryMonitor.lastRechargeableState;
			}
		}
		part emergencyController {
			attribute monitorsMeasurementFaults : Boolean = true;
			perform action respondToEmergency : ExecuteEmergencyResponse {
				in measurementFailure = not measurementOperational;
			}
		}
	}
	
	part glucoseMeterInstance : BloodGlucoseMeter {
		attribute measurementOperational = true;
	}
	
	requirement def 'BatteryDetectionRequirement' {
		text = "The system shall automatically detect depleted or unchargeable batteries and warn the user to avoid glucose measurement failures.";
	}
	
	requirement def 'AlarmAndResponseRequirement' {
		text = "Preventive battery measures plus alarms and emergency response shall activate when blood glucose measurement anomalies occur.";
	}
	
	requirement def 'RiskManagementRequirement' {
		text = "All related requirements must remain fully approved with managed high-risk scenarios so patients receive continuous therapeutic support.";
	}
}
