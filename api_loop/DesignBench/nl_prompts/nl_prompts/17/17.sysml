package 'VehicleFuelManagement17' {
	private import ScalarValues::*;

	part def VehicleFuelManagementSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part engine : Engine;
		attribute efficientFlowRequired : Boolean;
		attribute informationTrackingRequired : Boolean;
		attribute reliabilityRequirement : Boolean;
		attribute safetyRequirement : Boolean;
		attribute monitoringRequirement : Boolean;
	}

	part def FuelTankAssembly {
		port fuelSupplyPort : FuelInterfacePort;
		port fuelReturnPort : FuelInterfacePort;
	}

	part def Engine {
		port fuelInletPort : FuelInterfacePort;
		port fuelReturnPort : FuelInterfacePort;
	}

	port def FuelInterfacePort {
		attribute supplyFlow : FuelFlowMeasurement;
		attribute returnFlow : FuelFlowMeasurement;
		attribute fuelTemperature : FuelTemperatureTelemetry;
	}

	attribute def FuelFlowMeasurement {
		attribute flowRate : Real;
	}

	attribute def FuelTemperatureTelemetry {
		attribute measuredTemperature : Real;
	}

	connection def FuelSupplyPath {
		end fromTank : FuelTankAssembly::fuelSupplyPort;
		end toEngine : Engine::fuelInletPort;
	}

	connection def FuelReturnPath {
		end fromEngine : Engine::fuelReturnPort;
		end toTank : FuelTankAssembly::fuelReturnPort;
	}
}
