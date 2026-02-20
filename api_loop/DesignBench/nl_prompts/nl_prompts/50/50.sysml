package 'VehicleLifecycleManagement50' {
	private import ScalarValues::*;

	part def VehicleLifecycleManagementSystem {
		part vehicles : VehicleRecord[*];
		attribute assemblyTracked : Boolean;
		attribute deliveryTracked : Boolean;
		attribute ownershipTracked : Boolean;
		attribute drivingTracked : Boolean;
		attribute scrappingTracked : Boolean;
		attribute traceabilitySupported : Boolean;
		attribute inquiriesSupported : Boolean;
	}

	part def VehicleRecord {
		attribute assemblyStageRecorded : Boolean;
		attribute deliveryStageRecorded : Boolean;
		attribute deliveryDate : DateRecord;
		attribute ownershipHistory : OwnershipRecord[*];
		attribute drivingHistory : DrivingRecord[*];
		attribute scrappingInfo : ScrapRecord;
	}

	attribute def OwnershipRecord {
		attribute ownerIdentifier : OwnerInfo;
		attribute ownershipStageUnique : Boolean;
	}

	attribute def DrivingRecord {
		attribute driverIdentifier : DriverInfo;
		attribute drivingOperationSpecified : Boolean;
		attribute relatedOwnershipTraceable : Boolean;
	}

	attribute def ScrapRecord {
		attribute scrappingDocumented : Boolean;
	}

	attribute def DateRecord {
		attribute isoDate : String;
	}

	attribute def OwnerInfo {
		attribute name : String;
		attribute contact : String;
	}

	attribute def DriverInfo {
		attribute name : String;
		attribute licenseIdentifier : String;
	}
}
