package 'VehicleDriverStateTracking' {
	private import ScalarValues::*;
	
	part def Driver {
		attribute name : String;
	}
	
	part def Engine;
	
	part def VehicleState {
		attribute mass : Real;
		attribute gearPosition : String;
		attribute throttlePosition : Real;
		ref part driver : Driver;
	}
	
	part def Vehicle {
		attribute mass : Real;
		part operationalState {
			attribute gearPosition : String;
			attribute throttlePosition : Real;
		}
		part engine : Engine;
		ref part driver : Driver;
		part stateHistory : VehicleState[*];
	}
	
	part driverAlice : Driver {
		attribute name = "Alice";
	}
	
	part driverBob : Driver {
		attribute name = "Bob";
	}
	
	part vehicle : Vehicle {
		part engine;
		ref part driver :> driverAlice;
		part stateHistory {
			part initialSlice : VehicleState {
				attribute mass = 2000.0;
				attribute gearPosition = "neutral";
				attribute throttlePosition = 0.0;
				ref part driver :> driverAlice;
			}
			part secondSlice : VehicleState {
				attribute mass = 1500.0;
				attribute gearPosition = "second";
				attribute throttlePosition = 0.5;
				ref part driver :> driverBob;
			}
		}
	}
}
