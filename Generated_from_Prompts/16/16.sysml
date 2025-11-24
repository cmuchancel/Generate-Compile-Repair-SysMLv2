package 'VehicleFuelSupplyReturn' {
	attribute def Temp;
	item def Fuel;
	
	port def FuelOutletPort {
		attribute fuelTemperature : Temp;
		out item fuelSupply : Fuel;
		in item fuelReturn : Fuel;
	}
	
	port def FuelInletPort {
		attribute fuelTemperature : Temp;
		in item fuelSupply : Fuel;
		out item fuelReturn : Fuel;
	}
	
	part def FuelTankAssembly {
		port fuelOutlet : FuelOutletPort;
	}
	
	part def EngineUnit {
		port fuelInlet : FuelInletPort;
	}
	
	part fuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engineUnit : EngineUnit;
		connect fuelTankAssembly.fuelOutlet to engineUnit.fuelInlet;
	}
}
