package 'VehicleLifecycleManagement' {
	private import ScalarValues::*;
	private import Time::*;
	
	item def OwnerInfo {
		attribute name : String;
	}
	
	item def DriverInfo {
		attribute name : String;
	}
	
	part vehicleLifecycle {
		part assemblyStage {
			attribute isAssembled : Boolean;
		}
		part deliveryStage {
			attribute deliveredOn : Time::DateTime;
		}
		part ownershipHistory {
			part ownershipRecord[*] ordered {
				attribute owner : OwnerInfo;
				part driverRecord[*] ordered {
					attribute driver : DriverInfo;
					attribute drivingDate : Time::DateTime;
				}
			}
		}
		part scrappingStage {
			attribute isScrapped : Boolean;
			attribute scrappedOn : Time::DateTime;
		}
	}
}
