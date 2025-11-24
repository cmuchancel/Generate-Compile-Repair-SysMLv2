package 'VehicleHealthMonitoring' {
	private import ScalarValues::Real;
	private import Time::*;
	
	action def SendOverTemperatureWarning;
	action def ScheduleNextMaintenance { out nextMaintenanceTime : Time::DateTime; }
	
	part vehicleHealthManager {
		attribute currentTemperature : Real;
		attribute maxTemperature : Real;
		attribute nextMaintenanceTime : Time::DateTime;
		attribute maintenanceDurationHours : Real = 48.0;
		attribute maintenanceElapsedHours : Real;
		part vehicleController;
		perform action scheduleNextMaintenance : ScheduleNextMaintenance;
		perform action sendWarning : SendOverTemperatureWarning;
		
		state healthStates {
			entry action initial;
			transition initial then normal;
			state normal;
			transition 'normal-degraded'
				first normal
				accept when currentTemperature > maxTemperature
				do sendWarning
				then degraded;
			transition 'normal-maintenance'
				first normal
				accept at nextMaintenanceTime
				then maintenance;
			state degraded;
			transition 'degraded-normal'
				first degraded
				accept when currentTemperature <= maxTemperature
				then normal;
			transition 'degraded-maintenance'
				first degraded
				accept at nextMaintenanceTime
				then maintenance;
			state maintenance;
			transition 'maintenance-normal'
				first maintenance
				accept when maintenanceElapsedHours >= maintenanceDurationHours
				do scheduleNextMaintenance
				then normal;
		}
	}
}
