package 'VehicleMassAndEngineRequirements' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleMassRequirement {
		attribute fullFuelVehicleMass : MassValue;
		attribute emptyVehicleMass : MassValue;
		attribute fullFuelLimit : MassValue = 2000[kg];
		attribute emptyLimit : MassValue = 1500[kg];
		attribute fullFuelCompliant : Boolean default fullFuelVehicleMass <= fullFuelLimit;
		attribute emptyCompliant : Boolean default emptyVehicleMass <= emptyLimit;
	}
	part def EnginePerformanceRequirement {
		attribute engineSpeed : AngularVelocityValue;
		attribute requiredTorqueFromTable : TorqueValue;
		attribute producedTorque : TorqueValue;
		attribute conformsToPerformanceTable : Boolean default producedTorque == requiredTorqueFromTable;
		attribute performanceTableReference : String;
	}
	part def ClutchTransmissionCoupling {
		attribute engineOutputTorque : TorqueValue;
		attribute clutchTransmittedTorque : TorqueValue;
		attribute transmissionInputTorque : TorqueValue;
		attribute torqueTransferThroughClutch : Boolean default clutchTransmittedTorque == engineOutputTorque;
		attribute torqueDeliveredToTransmission : Boolean default transmissionInputTorque == clutchTransmittedTorque;
	}
	part def VehicleEngineSystem {
		part vehicleMassRequirement : VehicleMassRequirement;
		part enginePerformanceRequirement : EnginePerformanceRequirement;
		part clutchTransmissionCoupling : ClutchTransmissionCoupling {
			attribute :>> engineOutputTorque default enginePerformanceRequirement.producedTorque;
			attribute :>> clutchTransmittedTorque default enginePerformanceRequirement.producedTorque;
			attribute :>> transmissionInputTorque default enginePerformanceRequirement.producedTorque;
		}
		attribute requirementsSatisfied : Boolean default
			vehicleMassRequirement.fullFuelCompliant and
			vehicleMassRequirement.emptyCompliant and
			enginePerformanceRequirement.conformsToPerformanceTable and
			clutchTransmissionCoupling.torqueTransferThroughClutch and
			clutchTransmissionCoupling.torqueDeliveredToTransmission;
	}
}
