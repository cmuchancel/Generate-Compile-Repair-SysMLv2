package 'FuelSystemManagement' {
	attribute def Temp;
	item def Fuel;
	
	port def FuelSupplyPort {
		attribute temperature : Temp;
		out item fuelSupply : Fuel;
		in item fuelReturn : Fuel;
	}
	
	port def FuelInletPort {
		attribute temperature : Temp;
		in item fuelSupply : Fuel;
		out item fuelReturn : Fuel;
	}
	
	part def FuelTankAssembly {
		port supplyPort : FuelSupplyPort;
	}
	
	part def Engine {
		port inletPort : FuelInletPort;
	}
	
	part fuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engine : Engine;
		connect fuelTankAssembly.supplyPort to engine.inletPort;
	}
}
