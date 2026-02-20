package 'VehicleTimeSlices53' {
	private import ScalarValues::*;

	part def Vehicle {
		attribute mass : Real;
		attribute status : VehicleStatus;
		part engine : Engine;
		ref part driver : Driver;
	}

	attribute def VehicleStatus {
		attribute gearSetting : Integer;
		attribute throttlePosition : Real;
	}

	part def Engine;

	part def Driver {
		attribute name : String;
	}

	part def VehicleTimeSlice {
		attribute sliceLabel : String;
		attribute massSnapshot : Real;
		attribute gearSnapshot : Integer;
		attribute throttleSnapshot : Real;
		ref part driverSnapshot : Driver;
	}

	part def VehicleTimeline {
		part vehicle : Vehicle;
		part initialSlice : VehicleTimeSlice {
			sliceLabel = "initial";
			massSnapshot = 2000.0;
			gearSnapshot = 0;
			throttleSnapshot = 0.0;
			driverSnapshot = DriverAlice;
		}
		part nextSlice : VehicleTimeSlice {
			sliceLabel = "next";
			massSnapshot = 1500.0;
			gearSnapshot = 2;
			throttleSnapshot = 0.5;
			driverSnapshot = DriverBob;
		}
	}

	part DriverAlice : Driver {
		name = "Alice";
	}

	part DriverBob : Driver {
		name = "Bob";
	}

	part vehicle_1 : VehicleTimeline;
}
