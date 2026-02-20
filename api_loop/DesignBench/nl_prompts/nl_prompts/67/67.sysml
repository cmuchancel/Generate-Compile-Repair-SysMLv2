package 'VehicleMaintenanceScheduling' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::s;
	private import SI::h;
	part def MaintenanceState {
		attribute underMaintenance : Boolean;
		attribute healthy : Boolean;
	}
	part def VehicleMaintenancePlan {
		attribute nextMaintenanceTime : TimeValue;
		attribute maintenanceInterval : TimeValue;
		attribute maximumAllowedTemperature : TemperatureValue;
		attribute currentTime : TimeValue;
		attribute maintenanceDue : Boolean default currentTime >= nextMaintenanceTime;
		attribute status : MaintenanceState {
			attribute :>> underMaintenance default maintenanceDue;
			attribute :>> healthy default not maintenanceDue;
		}
	}
	part def MaintenanceExecutionRecord {
		attribute scheduledMaintenanceTime : TimeValue;
		attribute actualMaintenanceStartTime : TimeValue;
		attribute previousHealthyStateEndTime : TimeValue;
		attribute maintenanceDuration : TimeValue;
		attribute maintenanceCompletionTime : TimeValue;
		attribute startAfterSchedule : Boolean default actualMaintenanceStartTime > scheduledMaintenanceTime;
		attribute startAfterHealthyState : Boolean default actualMaintenanceStartTime >= previousHealthyStateEndTime;
		attribute startWithinTwoSeconds : Boolean default actualMaintenanceStartTime <= previousHealthyStateEndTime + 2[s];
		attribute durationWithinFortyEightHours : Boolean default maintenanceDuration <= 48[h];
	}
	part def VehicleMaintenanceSystem {
		part plan : VehicleMaintenancePlan;
		part execution : MaintenanceExecutionRecord {
			attribute :>> scheduledMaintenanceTime default plan.nextMaintenanceTime;
		}
		attribute nextMaintenanceTimeAfterService : TimeValue default execution.maintenanceCompletionTime + plan.maintenanceInterval;
		attribute healthRestored : MaintenanceState {
			attribute :>> underMaintenance default false;
			attribute :>> healthy default true;
		}
		attribute startCriteriaSatisfied : Boolean default
			execution.startAfterSchedule and
			execution.startAfterHealthyState and
			execution.startWithinTwoSeconds;
		attribute maintenanceDurationValid : Boolean default execution.durationWithinFortyEightHours;
		attribute maintenanceStatusEnforced : Boolean default
			(not plan.status.underMaintenance) or plan.maintenanceDue;
	}
}
