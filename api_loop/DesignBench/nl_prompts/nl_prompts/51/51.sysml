package 'VehicleStructure51' {
	private import ScalarValues::*;

	part def Vehicle {
		attribute mass : Real;
		attribute status : VehicleStatus;
		part engine : Engine;
		ref part driver : Driver;
		part wheels : Wheel[*];
	}

	attribute def VehicleStatus {
		attribute gearSetting : Integer;
		attribute acceleratorPosition : Real;
	}

	part def Engine;
	part def Driver;
	part def Wheel {
		attribute wheelIdentifier : String;
	}

part def VehicleInstance {
	part vehicle : Vehicle;
	part leftFrontWheel : Wheel {
		attribute wheelIdentifier : String = "left_front";
	}
	part rightFrontWheel : Wheel {
		attribute wheelIdentifier : String = "right_front";
	}
	part timeSnapshots : TimeSnapshot[*];
}

	part def TimeSnapshot {
		attribute snapshotLabel : String;
		part wheels : WheelSnapshot[*];
	}

	part def WheelSnapshot {
		attribute associatedWheel : String;
		attribute stateDescription : String;
	}
}
