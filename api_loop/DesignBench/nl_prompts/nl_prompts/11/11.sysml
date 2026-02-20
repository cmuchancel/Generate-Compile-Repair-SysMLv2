package 'MotorVehicleFuelSystem11' {
	private import ScalarValues::*;

	part def MotorVehicleFuelSystem {
		part fuelTank : FuelTankAssembly;
		part engine : Engine;
		attribute circulationProcess : FuelCirculationProcess;
		attribute monitoring : FuelFlowMonitoring;
		attribute stabilityRequirementSatisfied : Boolean;
		attribute reliabilityRequirementSatisfied : Boolean;
	}

	part def FuelTankAssembly {
		port fuelInterface : FuelSystemInterface;
	}

	part def Engine {
		port fuelInterface : FuelSystemInterface;
	}

	port def FuelSystemInterface {
		attribute supplyFlow : FuelFlowMeasurement;
		attribute returnFlow : FuelFlowMeasurement;
		attribute temperatureTelemetry : TemperatureMetric;
	}

	attribute def FuelFlowMeasurement {
		attribute flowValue : Real;
	}

	attribute def TemperatureMetric {
		attribute sensedTemperature : Real;
	}

	attribute def FuelFlowMonitoring {
		attribute monitoredTemperature : TemperatureMetric;
	}

	attribute def FuelCirculationProcess {
		attribute supplyStatus : FuelFlowMeasurement;
		attribute returnStatus : FuelFlowMeasurement;
	}

	connection def FuelSupplyRoute {
		end fromTank : FuelTankAssembly::fuelInterface;
		end toEngine : Engine::fuelInterface;
	}
}
