package 'FuelInterfaceManagement' {
	private import ScalarValues::*;
	part def Fuel;
	port def FuelSupplyInterface {
		attribute temperature : Real;
		out item fuelSupply : Fuel;
		in item fuelReturn : Fuel;
	}
	port def FuelReceiveInterface {
		attribute temperature : Real;
		in item fuelSupply : Fuel;
		out item fuelReturn : Fuel;
	}
	part def FuelTank {
		port supplyPort : FuelSupplyInterface;
	}
	part def Engine {
		port receivePort : FuelReceiveInterface;
	}
	part fuelFlowSystem {
		part tank : FuelTank;
		part engine : Engine;
		flow of Fuel from tank.supplyPort.fuelSupply to engine.receivePort.fuelSupply;
		flow of Fuel from engine.receivePort.fuelReturn to tank.supplyPort.fuelReturn;
	}
}
