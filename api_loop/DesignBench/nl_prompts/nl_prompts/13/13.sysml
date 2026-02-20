package 'VehicleFuelManagement13' {
	private import ScalarValues::*;

	part def VehicleFuelManagementSystem {
		part fuelSystem : FuelSystem;
		attribute temperatureTelemetry : FuelTemperatureTelemetry;
	}

	part def FuelSystem {
		part fuelTankAssembly : FuelTankAssembly;
		part returnLine : ReturnLine;
		part fuelConsumers : FuelConsumingUnit;
		attribute flowPathDefined : Boolean;
		attribute interfaceDefinitionsSpecified : Boolean;
		attribute fuelAndDataBindingEnforced : Boolean;
	}

	part def FuelTankAssembly {
		part fuelTank : FuelTank;
		part fuelPump : FuelPump;
		port pressurizedFuelPort : FuelFlowPort;
		port returnPort : ReturnFlowPort;
	}

	part def FuelTank;

	part def FuelPump {
		port dischargePort : FuelFlowPort;
		attribute pressurizationEnabled : Boolean;
	}

	part def FuelConsumingUnit {
		port fuelInlet : FuelFlowPort;
		port returnOutlet : ReturnFlowPort;
	}

	part def ReturnLine {
		port inlet : ReturnFlowPort;
		port outlet : ReturnFlowPort;
		attribute unusedFuelHandled : Boolean;
	}

	port def FuelFlowPort {
		attribute deliveredFuel : FuelFlowMeasurement;
		attribute temperatureData : FuelTemperatureTelemetry;
	}

	port def ReturnFlowPort {
		attribute returnedFuel : FuelFlowMeasurement;
	}

	attribute def FuelFlowMeasurement {
		attribute flowRate : Real;
	}

	attribute def FuelTemperatureTelemetry {
		attribute measuredTemperature : Real;
	}

	connection def PressurizedSupplyPath {
		end fromPump : FuelPump::dischargePort;
		end toConsumers : FuelConsumingUnit::fuelInlet;
	}

	connection def PumpToTankInterface {
		end interfacePort : FuelTankAssembly::pressurizedFuelPort;
		end pumpPort : FuelPump::dischargePort;
	}

	connection def ReturnFuelPath {
		end fromConsumers : FuelConsumingUnit::returnOutlet;
		end throughReturnLine : ReturnLine::inlet;
	}

	connection def ReturnLineToTank {
		end lineOutlet : ReturnLine::outlet;
		end tankPort : FuelTankAssembly::returnPort;
	}
}
