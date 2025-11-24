package 'VehicleMassFiltering' {
	private import ScalarValues::*;
	
	part def Component {
		attribute serialNumber : String;
		attribute mass : Real;
		part subcomponents : Component[*];
	}
	
	part def ComponentFilter {
		attribute threshold : Real;
		ref part filteredComponents : Component[*];
	}
	
	part def Engine :> Component;
	part def Transmission :> Component;
	
part def Vehicle {
		attribute vin : String;
		attribute baseMass : Real;
		attribute totalMass : Real;
		part engine : Engine;
		part transmission : Transmission;
		part body : Component;
		part interior : Component;
		part massFilter : ComponentFilter;
	}
	
	calc def VehicleMassRollup {
		in vehicle : Vehicle;
		return total : Real;
	}
	
	part vehicleMassManager {
		part vehicle : Vehicle {
			attribute vin = "VIN-0001";
			attribute baseMass = 800.0;
			attribute redefines totalMass = baseMass + engine.mass + transmission.mass + body.mass + interior.mass;
			part engine {
				attribute serialNumber = "ENG-001";
				attribute mass = 400.0;
			}
			part transmission {
				attribute serialNumber = "TRANS-001";
				attribute mass = 250.0;
			}
			part body {
				attribute serialNumber = "BODY-001";
				attribute mass = 350.0;
			}
			part interior {
				attribute serialNumber = "INT-001";
				attribute mass = 120.0;
			}
			part massFilter {
				attribute threshold = 100.0;
				ref part filteredComponents = (body, interior);
			}
		}
		calc rollup : VehicleMassRollup {
			in vehicle = vehicleMassManager.vehicle;
			return total;
		}
	}
}
