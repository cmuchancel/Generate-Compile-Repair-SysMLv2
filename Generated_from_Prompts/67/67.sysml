package 'VehicleMaintenanceScheduling' {
	private import ScalarValues::*;
	
	part vehicleMaintenance {
		attribute maxAllowedTemperature : Real;
		attribute nextMaintenanceTime : Real;
		attribute maintenanceInterval : Real;
		attribute currentTime : Real;
		attribute maintenanceStartTime : Real;
		attribute healthyStateEndTime : Real;
		attribute maintenanceDurationHours : Real;
		attribute nextMaintenanceTimeAfterMaintenance : Real = maintenanceStartTime + maintenanceInterval;
		attribute readyForMaintenance : Boolean = currentTime >= nextMaintenanceTime;
		attribute underMaintenance : Boolean = readyForMaintenance;
		constraint startAfterScheduled {
			constraint { maintenanceStartTime > nextMaintenanceTime }
		}
		constraint startWithinTwoSeconds {
			constraint { maintenanceStartTime - healthyStateEndTime <= 2.0 }
		}
		constraint durationLimit {
			constraint { maintenanceDurationHours <= 48.0 }
		}
	}
}
