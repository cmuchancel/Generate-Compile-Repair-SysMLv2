package 'SensorManagementSystem34' {
	private import ScalarValues::*;

	part def SensorManagementSystem {
		part sensorRegistry : SensorRegistry;
		attribute statusMonitoringProvided : Boolean;
		attribute readyDetectionImplemented : Boolean;
		attribute automaticTraversalProvided : Boolean;
		attribute sensorsKeptUpToDate : Boolean;
		attribute systemResponsivenessEnhanced : Boolean;
		attribute systemReliabilityEnhanced : Boolean;
	}

	part def SensorRegistry {
		part sensors : ManagedSensor[*];
		attribute registeredSensorsMaintained : Boolean;
	}

	part def ManagedSensor {
		attribute status : SensorStatus;
		attribute updateOperationPerformed : Boolean;
	}

	attribute def SensorStatus {
		attribute readyState : Boolean;
		attribute lastUpdatedTime : TimeValue;
	}

	attribute def TimeValue {
		attribute timeStamp : Real;
	}
}
