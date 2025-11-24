package 'VehicleStateEvolution' {
	private import ScalarValues::*;
	
	part def Driver {
		attribute name : String;
	}
	
	part def Engine {
		attribute identifier : String;
	}
	
	part def VehicleState {
		attribute mass : Real;
		attribute gearSetting : Integer;
		attribute acceleratorPedalPosition : Real;
	}
	
	part def Vehicle {
		attribute mass : Real;
		part operationalState {
			attribute gearSetting : Integer;
			attribute acceleratorPedalPosition : Real;
		}
		part engine : Engine;
		ref part driver : Driver;
		part states : VehicleState[*];
	}
	
	part driver_1 : Driver {
		attribute name = "Driver One";
	}
	
part vehicle_1 : Vehicle {
		part engine : Engine {
			attribute identifier = "Engine-A";
		}
		ref part driver :> driver_1;
		part states {
			part initialState : VehicleState {
				attribute mass = 2000.0;
				attribute gearSetting = 0;
				attribute acceleratorPedalPosition = 0.0;
			}
			part nextState : VehicleState {
				attribute mass = 1500.0;
				attribute gearSetting = 2;
				attribute acceleratorPedalPosition = 0.5;
			}
		}
	}
}
