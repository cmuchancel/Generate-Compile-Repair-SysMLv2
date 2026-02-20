package 'VehicleMassLimitAndTorqueTransmission' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleMassEvaluation {
		attribute curbMass : MassValue;
		attribute fuelMassFullTank : MassValue;
		attribute fuelMassEmptyTank : MassValue;
		attribute fullTankLimit : MassValue = 2000[kg];
		attribute emptyTankLimit : MassValue = 1500[kg];
		attribute fullTankTotalMass : MassValue default curbMass + fuelMassFullTank;
		attribute emptyTankTotalMass : MassValue default curbMass + fuelMassEmptyTank;
		attribute fullTankCompliant : Boolean default fullTankTotalMass <= fullTankLimit;
		attribute emptyTankCompliant : Boolean default emptyTankTotalMass <= emptyTankLimit;
	}
	part def EngineTorqueCharacteristic {
		attribute engineSpeed : AngularVelocityValue;
		attribute specifiedTorque : TorqueValue;
		attribute actualTorque : TorqueValue;
		attribute matchesSpecification : Boolean default actualTorque == specifiedTorque;
	}
	part def ClutchGearboxInterface {
		attribute torqueFromEngine : TorqueValue;
		attribute torqueThroughClutch : TorqueValue;
		attribute gearboxInputTorque : TorqueValue;
		attribute torqueTransmitted : Boolean default torqueThroughClutch == torqueFromEngine;
		attribute smoothPowerDelivery : Boolean default gearboxInputTorque == torqueThroughClutch;
	}
	part def VehiclePowertrainCompliance {
		part massEvaluation : VehicleMassEvaluation;
		part torqueCharacteristic : EngineTorqueCharacteristic;
		part clutchInterface : ClutchGearboxInterface {
			attribute :>> torqueFromEngine default torqueCharacteristic.actualTorque;
			attribute :>> torqueThroughClutch default torqueCharacteristic.actualTorque;
			attribute :>> gearboxInputTorque default torqueCharacteristic.actualTorque;
		}
		attribute requirementsMet : Boolean default
			massEvaluation.fullTankCompliant and
			massEvaluation.emptyTankCompliant and
			torqueCharacteristic.matchesSpecification and
			clutchInterface.torqueTransmitted and
			clutchInterface.smoothPowerDelivery;
	}
}
