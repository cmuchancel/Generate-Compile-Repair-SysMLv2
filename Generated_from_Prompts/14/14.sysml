package 'AutomotiveFuelSystem' {
	attribute def Temp;
	item def Fuel;
	
	port def FuelOutPort {
		attribute temperature : Temp;
		out item fuelSupply : Fuel;
		in item fuelReturn : Fuel;
	}
	
	port def FuelInPort {
		attribute temperature : Temp;
		in item fuelSupply : Fuel;
		out item fuelReturn : Fuel;
	}
	
	part def FuelTank;
	
	part def FuelPump {
		port fuelPort : FuelOutPort;
	}
	
	part def FuelTankAssembly {
		part fuelTank : FuelTank;
		part fuelPump : FuelPump;
		port fuelOutlet : FuelOutPort;
		connect fuelOutlet to fuelPump.fuelPort;
	}
	
	part def Engine {
		port fuelPort : FuelInPort;
	}
	
	part fuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engine : Engine;
		connect fuelTankAssembly.fuelOutlet to engine.fuelPort;
	}
}
