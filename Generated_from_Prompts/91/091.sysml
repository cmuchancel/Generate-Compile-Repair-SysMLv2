package 'EquipmentRiskMonitoring' {
	private import ScalarValues::*;
	enum def SeverityLevel {
		enum low;
		enum medium;
		enum high;
	}
	part def Battery {
		attribute level : Real;
		attribute minimumLevel : Real;
		attribute lowBatteryDuration : Real;
		attribute allowedLowBatteryDuration : Real;
	}
	part def Equipment {
		part battery : Battery;
		attribute criticalFunctionActive : Boolean;
	}
	action def MonitorBattery {
		in battery : Battery;
		out level : Real;
	}
	constraint def LowBatteryDetection {
		in battery : Battery;
		assert constraint { battery.level < battery.minimumLevel };
	}
	constraint def ShutdownTrigger {
		in battery : Battery;
		assert constraint { battery.lowBatteryDuration >= battery.allowedLowBatteryDuration };
	}
	occurrence def RiskSituation;
	occurrence def RiskCause :> RiskSituation {
		attribute probability : Real;
	}
	occurrence def RiskCondition :> RiskSituation;
	occurrence def FailureConsequence :> RiskSituation {
		attribute severityLevel : SeverityLevel;
	}
	connection def ScenarioPropagation {
		end precursor : RiskSituation[1];
		end result : RiskSituation[1];
	}
	part def RiskScenario {
		occurrence situations : RiskSituation[*];
		connection propagations : ScenarioPropagation[*];
	}
	part riskRegister {
		part equipment : Equipment;
		occurrence batteryAging : RiskCause;
		occurrence lowBatteryLevel : RiskCondition;
		occurrence lowBatteryState : RiskCondition;
		occurrence deviceShutdown : FailureConsequence {
			attribute redefines severityLevel = SeverityLevel::high;
		}
		action monitorBatteryLevel : MonitorBattery {
			in battery = equipment.battery;
		}
		connection agingCausesLevel : ScenarioPropagation {
			end precursor = batteryAging;
			end result = lowBatteryLevel;
		}
		connection levelLeadsState : ScenarioPropagation {
			end precursor = lowBatteryLevel;
			end result = lowBatteryState;
		}
		connection persistentLowBattery : ScenarioPropagation {
			end precursor = lowBatteryState;
			end result = deviceShutdown;
		}
		part compositeRisk : RiskScenario {
			occurrence redefines situations = (batteryAging, lowBatteryLevel, lowBatteryState, deviceShutdown);
			connection redefines propagations = (agingCausesLevel, levelLeadsState, persistentLowBattery);
		}
		assert constraint detectLowBattery : LowBatteryDetection {
			in battery = equipment.battery;
		}
		assert constraint autoShutdown : ShutdownTrigger {
			in battery = equipment.battery;
		}
	}
}
