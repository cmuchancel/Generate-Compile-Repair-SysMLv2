package 'VehicleStructureTracking' {
	private import ScalarValues::*;
	private import Time::*;
	
	part def Driver {
		attribute name : String;
	}
	
	part def Engine {
		attribute serialNumber : String;
	}
	
	part def Wheel {
		attribute position : String;
	}
	
	part def WheelSnapshot {
		attribute timestamp : Time::DateTime;
		attribute statusDescription : String;
		ref part wheel : Wheel;
	}
	
	part def Vehicle {
		attribute mass : Real;
		part operationalStatus {
			attribute gearSetting : String;
			attribute acceleratorPedalPosition : String;
		}
		part engine : Engine;
		ref part driver : Driver;
		part leftFrontWheel : Wheel {
			attribute redefines position = "left front";
		}
		part rightFrontWheel : Wheel {
			attribute redefines position = "right front";
		}
		part wheelSnapshots : WheelSnapshot[*];
	}
	
	part driverRegistry {
		part driver_1 : Driver {
			attribute name = "Driver A";
		}
	}
	
	part vehicle_1 : Vehicle {
		attribute mass = 1500.0;
		ref part driver :> driverRegistry.driver_1;
		part wheelSnapshots {
			part leftFront_t1 : WheelSnapshot {
				ref part wheel :> leftFrontWheel;
				attribute statusDescription = "Pre-delivery inspection";
			}
			part rightFront_t2 : WheelSnapshot {
				ref part wheel :> rightFrontWheel;
				attribute statusDescription = "Post-maintenance check";
			}
		}
	}
}
