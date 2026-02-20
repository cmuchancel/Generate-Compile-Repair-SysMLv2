package 'VehicleMassAndPowerDelivery' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleMassProfile {
		attribute curbMass : MassValue;
		attribute fuelMass : MassValue;
		attribute maximumAllowedMass : MassValue;
		attribute totalMass : MassValue default curbMass + fuelMass;
		attribute fuelMassPositive : Boolean default fuelMass > 0[kg];
		attribute withinLimit : Boolean default totalMass <= maximumAllowedMass;
	}
	part def EngineTorqueCurveRequirement {
		attribute engineSpeed : AngularVelocityValue;
		attribute torqueFromSpeedCurve : TorqueValue;
		attribute producedTorque : TorqueValue;
		attribute meetsSpeedCurve : Boolean default producedTorque == torqueFromSpeedCurve;
	}
	part def ClutchInterface {
		attribute inputTorque : TorqueValue;
		attribute outputTorque : TorqueValue;
		attribute torqueTransmissionEffective : Boolean default outputTorque == inputTorque;
	}
	part def TransmissionInput {
		attribute receivedTorque : TorqueValue;
	}
	part def PowerDeliverySystem {
		part massProfile : VehicleMassProfile;
		part torqueRequirement : EngineTorqueCurveRequirement;
		part clutch : ClutchInterface {
			attribute :>> inputTorque default torqueRequirement.producedTorque;
		}
		part transmission : TransmissionInput {
			attribute :>> receivedTorque default clutch.outputTorque;
		}
		attribute smoothPowerDelivery : Boolean default
			clutch.torqueTransmissionEffective and
			transmission.receivedTorque == torqueRequirement.producedTorque;
	}
}
