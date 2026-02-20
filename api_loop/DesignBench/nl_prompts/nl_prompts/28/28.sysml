package 'BatteryChargingControl28' {
	private import ScalarValues::*;

	part def BatteryChargingControlSystem {
		part battery : Battery;
		part levelDisplay : BatteryLevelDisplay;
		part monitoringController : MonitoringController;
		part chargingUnit : ChargingUnit;
		attribute continuousMonitoringPerformed : Boolean;
		attribute automatedReplenishmentProvided : Boolean;
		attribute automatedTerminationProvided : Boolean;
		attribute manualInterventionNotRequired : Boolean;
		attribute overchargePreventionProvided : Boolean;
		attribute safeFullChargeEnsured : Boolean;
		attribute reliableFullChargeEnsured : Boolean;
	}

	part def Battery {
		port statePort : StatePort;
	}

	part def BatteryLevelDisplay {
		port displayedState : StatePort;
		attribute percentageShown : Boolean;
	}

	part def MonitoringController {
		port monitoredState : StatePort;
		port replenishCommand : ChargingCommandPort;
		attribute replenishmentLoopActive : Boolean;
	}

	part def ChargingUnit {
		port commandInput : ChargingCommandPort;
		port chargingOutput : ChargingPort;
	}

	port def StatePort {
		attribute stateOfChargePercentage : StateOfChargePercentage;
	}

	port def ChargingCommandPort {
		attribute chargingEnabled : Boolean;
	}

	port def ChargingPort {
		attribute deliveredCharge : ChargeRate;
	}

	attribute def StateOfChargePercentage {
		attribute percentageValue : Real;
	}

	attribute def ChargeRate {
		attribute rateValue : Real;
	}

	connection def BatteryToDisplay {
		end fromBattery : Battery::statePort;
		end toDisplay : BatteryLevelDisplay::displayedState;
	}

	connection def BatteryToController {
		end fromBattery : Battery::statePort;
		end toController : MonitoringController::monitoredState;
	}

	connection def ControllerToChargingUnit {
		end fromController : MonitoringController::replenishCommand;
		end toUnit : ChargingUnit::commandInput;
	}

	connection def ChargingUnitToBattery {
		end fromUnit : ChargingUnit::chargingOutput;
		end toBattery : Battery::statePort;
	}
}
