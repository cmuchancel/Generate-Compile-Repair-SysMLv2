package 'VehicleHealthMonitoring40' {
	private import ScalarValues::*;

	part def VehicleHealthMonitoringSystem {
		part temperatureMonitor : TemperatureMonitor;
		part healthStatusManager : HealthStatusManager;
		part maintenanceScheduler : MaintenanceScheduler;
		part controllerNotification : ControllerNotification;
		attribute automaticOperationProvided : Boolean;
		attribute safetyEnhanced : Boolean;
		attribute reliabilityEnhanced : Boolean;
		attribute manualInterventionReduced : Boolean;
	}

	part def TemperatureMonitor {
		attribute currentTemperature : TemperatureValue;
		attribute maxAllowableTemperature : TemperatureValue;
		attribute temperatureSafeRange : Boolean;
		attribute overTemperatureDetected : Boolean;
	}

	part def HealthStatusManager {
		attribute statusNormal : Boolean;
		attribute statusDegraded : Boolean;
		attribute maintenanceStatus : Boolean;
		attribute degradedModeActive : Boolean;
		attribute automaticStatusReversion : Boolean;
	}

	part def MaintenanceScheduler {
		attribute maintenanceInterval : TimeIntervalValue;
		attribute maintenanceStatusDurationHours : DurationValue;
		attribute maintenanceStatusDuration48Hours : Boolean;
		attribute maintenanceTimeReached : Boolean;
		attribute nextMaintenanceTime : TimeValue;
		attribute maintenanceCompleted : Boolean;
	}

	part def ControllerNotification {
		attribute overTemperatureWarningSent : Boolean;
	}

	attribute def TemperatureValue {
		attribute temperature : Real;
	}

	attribute def TimeIntervalValue {
		attribute intervalHours : Real;
	}

	attribute def DurationValue {
		attribute hours : Real;
	}

	attribute def TimeValue {
		attribute timestamp : Real;
	}
}
