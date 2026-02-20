package 'FuelCirculationSystem16' {
	private import ScalarValues::*;

	part def FuelCirculationSystem {
		part fuelTank : FuelTankAssembly;
		part engine : EngineUnit;
		attribute continuousFuelSupply : Boolean;
		attribute reliableFuelSupply : Boolean;
		attribute fuelReturnProcessingSupported : Boolean;
		attribute managementIntelligenceImproved : Boolean;
		attribute managementSafetyImproved : Boolean;
	}

	part def FuelTankAssembly {
		port fuelOutletPort : FuelInterfacePort;
		port fuelReturnPort : FuelInterfacePort;
	}

	part def EngineUnit {
		port fuelInletPort : FuelInterfacePort;
		port fuelReturnPort : FuelInterfacePort;
	}

	port def FuelInterfacePort {
		attribute fuelSupplyFlow : FuelFlowMeasurement;
		attribute fuelReturnFlow : FuelFlowMeasurement;
		attribute fuelTemperature : FuelTemperatureTelemetry;
	}

	attribute def FuelFlowMeasurement {
		attribute flowRate : Real;
	}

	attribute def FuelTemperatureTelemetry {
		attribute measuredTemperature : Real;
	}

	connection def OutletToEngineInterface {
		end fromTank : FuelTankAssembly::fuelOutletPort;
		end toEngine : EngineUnit::fuelInletPort;
	}

	connection def EngineReturnInterface {
		end fromEngine : EngineUnit::fuelReturnPort;
		end toTank : FuelTankAssembly::fuelReturnPort;
	}

	connection def BidirectionalInterface {
		end tankInterface : FuelTankAssembly::fuelOutletPort;
		end engineInterface : EngineUnit::fuelReturnPort;
	}
}
