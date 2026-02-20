package 'BatteryChargingControl27' {
	private import ScalarValues::*;

	part def BatteryChargingControlSystem {
		part battery : Battery;
		part levelMonitor : BatteryLevelMonitor;
		part chargingController : ChargingController;
		part chargingSource : ChargingSource;
		attribute safeChargingEnsured : Boolean;
		attribute efficientChargingEnsured : Boolean;
		attribute fullChargeTarget : Boolean;
		attribute healthyChargingMaintained : Boolean;
		attribute convenienceEnhanced : Boolean;
		attribute safetyEnhanced : Boolean;
	}

	part def Battery {
		port stateOfChargePort : StateOfChargePort;
	}

	part def BatteryLevelMonitor {
		port monitoredState : StateOfChargePort;
		attribute realTimeMonitoring : Boolean;
	}

	part def ChargingController {
		port levelInput : StateOfChargePort;
		port chargingCommand : ChargingCommandPort;
		attribute automaticChargingSupplied : Boolean;
		attribute automaticTerminationPerformed : Boolean;
	}

	part def ChargingSource {
		port commandInput : ChargingCommandPort;
		port chargingOutput : ChargingOutputPort;
	}

	port def StateOfChargePort {
		attribute stateOfCharge : ChargeFraction;
	}

	port def ChargingCommandPort {
		attribute enableCharging : Boolean;
	}

	port def ChargingOutputPort {
		attribute deliveredCurrent : CurrentMeasurement;
	}

	attribute def ChargeFraction {
		attribute percentage : Real;
	}

	attribute def CurrentMeasurement {
		attribute currentValue : Real;
	}

	connection def BatteryToMonitor {
		end fromBattery : Battery::stateOfChargePort;
		end toMonitor : BatteryLevelMonitor::monitoredState;
	}

	connection def MonitorToController {
		end fromMonitor : BatteryLevelMonitor::monitoredState;
		end toController : ChargingController::levelInput;
	}

	connection def ControllerToSource {
		end fromController : ChargingController::chargingCommand;
		end toSource : ChargingSource::commandInput;
	}

	connection def SourceToBattery {
		end fromSource : ChargingSource::chargingOutput;
		end toBattery : Battery::stateOfChargePort;
	}
}
