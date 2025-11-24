package 'VehicleFuelSystem' {
	attribute def Temp;
	item def Fuel;
	
	port def FuelSupplyPort {
		attribute temperature : Temp;
		out item fuelSupply : Fuel;
		in item fuelReturn : Fuel;
	}
	
	port def FuelReceivePort {
		attribute temperature : Temp;
		in item fuelSupply : Fuel;
		out item fuelReturn : Fuel;
	}
	
	part def FuelTankAssembly {
		port fuelPort : FuelSupplyPort;
	}
	
	part def Engine {
		port fuelPort : FuelReceivePort;
	}
	
	part fuelFlowSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engine : Engine;
		connect fuelTankAssembly.fuelPort to engine.fuelPort;
	}
}
