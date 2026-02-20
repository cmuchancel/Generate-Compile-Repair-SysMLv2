package 'VehicleMassAndPowertrainControl' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleMassLimits {
		attribute dryMass : MassValue;
		attribute fuelMass : MassValue;
		attribute fullTankLimit : MassValue = 2000[kg];
		attribute emptyTankLimit : MassValue = 1500[kg];
		attribute fullTankTotalMass : MassValue default dryMass + fuelMass;
		attribute emptyTankTotalMass : MassValue default dryMass;
		attribute withinFullTankLimit : Boolean default fullTankTotalMass <= fullTankLimit;
		attribute withinEmptyTankLimit : Boolean default emptyTankTotalMass <= emptyTankLimit;
	}
	part def EngineSpeedTorqueRequirement {
		attribute speed : AngularVelocityValue;
		attribute torqueFromCurve : TorqueValue;
		attribute producedTorque : TorqueValue;
		attribute meetsCurve : Boolean default producedTorque == torqueFromCurve;
		attribute curveReference : String;
	}
	part def ClutchToTransmissionInterface {
		attribute clutchInputTorque : TorqueValue;
		attribute clutchOutputTorque : TorqueValue;
		attribute transmissionInputTorque : TorqueValue;
		attribute torqueTransmitted : Boolean default clutchOutputTorque == clutchInputTorque;
		attribute torqueDeliveredToTransmission : Boolean default transmissionInputTorque == clutchOutputTorque;
	}
	part def VehiclePowertrainSystem {
		part massLimits : VehicleMassLimits;
		part torqueRequirement : EngineSpeedTorqueRequirement;
		part clutchInterface : ClutchToTransmissionInterface {
			attribute :>> clutchInputTorque default torqueRequirement.producedTorque;
			attribute :>> clutchOutputTorque default torqueRequirement.producedTorque;
			attribute :>> transmissionInputTorque default torqueRequirement.producedTorque;
		}
		attribute smoothPowerDelivery : Boolean default
			clutchInterface.torqueTransmitted and
			clutchInterface.torqueDeliveredToTransmission;
		attribute meetsAllRequirements : Boolean default
			massLimits.withinFullTankLimit and
			massLimits.withinEmptyTankLimit and
			torqueRequirement.meetsCurve and
			smoothPowerDelivery;
	}
}
