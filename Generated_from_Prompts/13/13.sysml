package 'VehicleFuelManagement' {
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
	
	part def FuelTank;
	
	part def FuelPump {
		port supplyPort : FuelSupplyPort;
	}
	
	part def FuelTankAssembly {
		part fuelTank : FuelTank;
		part fuelPump : FuelPump;
		port fuelLine : FuelSupplyPort;
		connect fuelLine to fuelPump.supplyPort;
	}
	
	part def FuelConsumer {
		port fuelInlet : FuelReceivePort;
	}
	
	part fuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part fuelConsumer : FuelConsumer;
		connect fuelTankAssembly.fuelLine to fuelConsumer.fuelInlet;
	}
}
