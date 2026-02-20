package 'VehicleFuelEconomySimulation' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleParameters {
		attribute curbWeight : MassValue;
		attribute load : MassValue;
		attribute wheelDiameter : LengthValue;
		attribute transmissionEfficiency : Real;
		attribute totalVehicleMass : MassValue default curbWeight + load;
	}
	part def DriveCyclePoint {
		attribute time : TimeValue;
		attribute position : LengthValue;
		attribute speed : SpeedValue;
	}
	part def DriveCycle {
		attribute scenarioType : String;
		part points : DriveCyclePoint[*];
	}
part def WheelPowerAnalysis {
	attribute vehicleMassInput : MassValue;
	part driveCycle : DriveCycle;
	attribute requiredWheelPower : PowerValue;
	attribute accelerationProfileAvailable : Boolean;
	attribute energyConsumption : EnergyValue;
}
part def EnginePowerDemandEstimate {
	attribute referencedWheelPower : PowerValue;
	attribute enginePowerDemand : PowerValue;
}
	part def FuelEconomyResult {
		attribute milesPerGallon : Real;
		attribute litersPer100Kilometers : Real;
		attribute meetsThirtyMpgRequirement : Boolean default milesPerGallon > 30.0;
	}
part def ScenarioFuelEconomyAssessment {
	part driveCycle : DriveCycle;
	part wheelPowerAnalysis : WheelPowerAnalysis;
	part enginePowerDemandEstimate : EnginePowerDemandEstimate {
		attribute :>> referencedWheelPower default wheelPowerAnalysis.requiredWheelPower;
	}
	part fuelEconomyResult : FuelEconomyResult;
}
part def VehicleFuelEconomySystem {
	part parameters : VehicleParameters;
	part urbanAssessment : ScenarioFuelEconomyAssessment {
		attribute :>> wheelPowerAnalysis.vehicleMassInput default parameters.totalVehicleMass;
	}
	part highwayAssessment : ScenarioFuelEconomyAssessment {
		attribute :>> wheelPowerAnalysis.vehicleMassInput default parameters.totalVehicleMass;
	}
		attribute complianceVerified : Boolean default
			urbanAssessment.fuelEconomyResult.meetsThirtyMpgRequirement and
			highwayAssessment.fuelEconomyResult.meetsThirtyMpgRequirement;
		attribute fuelEconomyPerHundredKilometersReported : Boolean default
			urbanAssessment.fuelEconomyResult.litersPer100Kilometers > 0.0 and
			highwayAssessment.fuelEconomyResult.litersPer100Kilometers > 0.0;
	}
}
