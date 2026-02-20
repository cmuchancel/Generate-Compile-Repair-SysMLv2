package 'CompositeVehicleSystem52' {
	private import ScalarValues::*;

	part def Vehicle {
		attribute mass : Real;
		attribute status : VehicleStatus;
		part engine : Engine;
		ref part driver : Driver;
	}

	attribute def VehicleStatus {
		attribute gearSetting : Integer;
		attribute acceleratorPosition : Real;
	}

	part def Engine;
	part def Driver;

part def VehicleTimeline {
	part vehicle : Vehicle;
	attribute initialSnapshotLabel : String = "initial";
	attribute initialMass : Real = 2000.0;
	attribute initialGearSetting : Integer = 0;
	attribute initialAcceleratorPosition : Real = 0.0;
	attribute nextSnapshotLabel : String = "next";
	attribute nextMass : Real = 1500.0;
	attribute nextGearSetting : Integer = 2;
	attribute nextAcceleratorPosition : Real = 0.5;
}

part vehicle_1 : VehicleTimeline;
}
