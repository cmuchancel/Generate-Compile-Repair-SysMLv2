package 'VehicleFuelSystem15' {
	private import ScalarValues::*;

	part def VehicleFuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engine : Engine;
		part supplyLine : FuelLine;
		part returnLine : FuelLine;
		attribute temperatureMonitoring : FuelTemperatureTelemetry;
		attribute continuousSupplyMaintained : Boolean;
		attribute stableSupplyMaintained : Boolean;
		attribute excessFuelRecycled : Boolean;
		attribute resourceUtilizationOptimized : Boolean;
	}

	part def FuelTankAssembly {
		port supplyPort : FuelLinePort;
		port returnPort : FuelLinePort;
		attribute fuelSupplyRole : Boolean;
	}

	part def Engine {
		port fuelInlet : FuelLinePort;
		port fuelReturn : FuelLinePort;
	}

	part def FuelLine {
		port startPort : FuelLinePort;
		port endPort : FuelLinePort;
	}

	port def FuelLinePort {
		attribute fuelFlow : FuelFlowMeasurement;
		attribute temperatureData : FuelTemperatureTelemetry;
	}

	attribute def FuelFlowMeasurement {
		attribute flowRate : Real;
	}

	attribute def FuelTemperatureTelemetry {
		attribute measuredTemperature : Real;
	}

	connection def SupplyConnection {
		end fromTank : FuelTankAssembly::supplyPort;
		end toEngine : Engine::fuelInlet;
	}

	connection def ReturnConnection {
		end fromEngine : Engine::fuelReturn;
		end toTank : FuelTankAssembly::returnPort;
	}

	connection def SupplyLineFromTank {
		end tankPort : FuelTankAssembly::supplyPort;
		end lineStart : FuelLine::startPort;
	}

	connection def SupplyLineToEngine {
		end lineEnd : FuelLine::endPort;
		end enginePort : Engine::fuelInlet;
	}

	connection def ReturnLineFromEngine {
		end engineReturn : Engine::fuelReturn;
		end lineInlet : FuelLine::startPort;
	}

	connection def ReturnLineToTank {
		end lineOutlet : FuelLine::endPort;
		end tankReturn : FuelTankAssembly::returnPort;
	}
}
