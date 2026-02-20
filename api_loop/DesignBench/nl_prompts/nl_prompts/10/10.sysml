package 'FuelFlowManagement10' {
	private import ScalarValues::*;

	part def FuelFlowManagementSystem {
		part fuelTank : FuelTankAssembly;
		part engine : Engine;
		attribute userDataAccess : UserFuelFlowData;
	}

	part def FuelTankAssembly {
		port dedicatedFuelPort : FuelTransferPort;
	}

	part def Engine {
		port fuelManagementPort : FuelTransferPort;
	}

	port def FuelTransferPort {
		attribute fuelSupplyFlow : FlowMeasurement;
		attribute fuelReturnFlow : FlowMeasurement;
		attribute interfaceTemperature : TemperatureTelemetry;
	}

	attribute def FlowMeasurement {
		attribute measuredFlow : Real;
	}

	attribute def TemperatureTelemetry {
		attribute measuredTemperature : Real;
	}

	attribute def UserFuelFlowData {
		attribute supplyData : FlowMeasurement;
		attribute returnData : FlowMeasurement;
		attribute temperatureParameters : TemperatureTelemetry;
	}

	connection def FuelTransferCircuit {
		end fromTank : FuelTankAssembly::dedicatedFuelPort;
		end toEngine : Engine::fuelManagementPort;
	}
}
