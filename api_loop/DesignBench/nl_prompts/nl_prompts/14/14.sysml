package 'AutomotiveFuelSystem14' {
	private import ScalarValues::*;

	part def AutomotiveFuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engine : Engine;
		attribute temperatureMonitoring : FuelTemperatureTelemetry;
		attribute efficientCirculationRequired : Boolean;
		attribute safeCirculationRequired : Boolean;
		attribute normalEngineSupplyEnsured : Boolean;
	}

	part def FuelTankAssembly {
		part fuelPump : FuelPump;
		part fuelTank : FuelTank;
		port outletInterface : FuelCirculationPort;
	}

	part def FuelPump {
		port circulationPort : FuelCirculationPort;
		attribute installedInsideAssembly : Boolean;
	}

	part def FuelTank;

	part def Engine {
		port fuelPort : FuelCirculationPort;
	}

	port def FuelCirculationPort {
		attribute supplyFlow : FuelFlowMeasurement;
		attribute returnFlow : FuelFlowMeasurement;
		attribute fuelTemperatureData : FuelTemperatureTelemetry;
	}

	attribute def FuelFlowMeasurement {
		attribute flowRate : Real;
	}

	attribute def FuelTemperatureTelemetry {
		attribute measuredTemperature : Real;
	}

	connection def AssemblyToEngineFlow {
		end fromAssembly : FuelTankAssembly::outletInterface;
		end toEngine : Engine::fuelPort;
	}

	connection def PumpToAssemblyFlow {
		end pumpPort : FuelPump::circulationPort;
		end assemblyOutlet : FuelTankAssembly::outletInterface;
	}
}
