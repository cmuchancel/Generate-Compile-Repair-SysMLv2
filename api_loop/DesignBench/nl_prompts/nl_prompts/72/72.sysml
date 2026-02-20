package 'VehicleDynamicPerformanceAnalysis' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleConfiguration {
		attribute curbWeight : MassValue;
		attribute payload : MassValue;
		attribute wheelDiameter : LengthValue;
		attribute transmissionEfficiency : Real;
		attribute urbanFuelEconomyMpg : Real;
		attribute highwayFuelEconomyMpg : Real;
		attribute totalVehicleMass : MassValue default curbWeight + payload;
	}
	part def DriveCyclePoint {
		attribute time : TimeValue;
		attribute position : LengthValue;
		attribute speed : SpeedValue;
	}
	part def DriveCycle {
		attribute name : String;
		part points : DriveCyclePoint[*];
	}
	part def DriveCycleResultPoint {
		attribute time : TimeValue;
		attribute position : LengthValue;
		attribute speed : SpeedValue;
		attribute acceleration : AccelerationValue;
		attribute powerDemand : PowerValue;
	}
	part def DynamicAnalysisResult {
		part resultPoints : DriveCycleResultPoint[*];
		attribute peakPowerDemand : PowerValue;
		attribute peakAcceleration : AccelerationValue;
	}
	part def EnergyConsumptionTarget {
		attribute targetMilesPerGallon : Real;
		attribute targetFuelPer100Kilometers : Real;
	}
part def FuelEconomyAssessment {
	attribute cycleFuelEconomyMpg : Real;
	attribute cycleFuelConsumptionPer100Kilometers : Real;
}
	part def AnalysisConfiguration {
		part vehicleConfiguration : VehicleConfiguration;
		part driveCycle : DriveCycle;
		attribute adjustableParametersProvided : Boolean;
	}
	part def VehicleDynamicAnalysisSystem {
		part configuration : AnalysisConfiguration;
	part dynamicResults : DynamicAnalysisResult;
	part fuelEconomyAssessment : FuelEconomyAssessment;
	part energyTarget : EnergyConsumptionTarget;
	attribute requiredEnginePower : PowerValue;
	attribute requiredAcceleration : AccelerationValue;
	attribute fuelEconomyVerified : Boolean default
		(fuelEconomyAssessment.cycleFuelEconomyMpg >= energyTarget.targetMilesPerGallon) or
		(fuelEconomyAssessment.cycleFuelConsumptionPer100Kilometers <= energyTarget.targetFuelPer100Kilometers);
		attribute powerAndAccelerationCovered : Boolean default
			requiredEnginePower == dynamicResults.peakPowerDemand and
			requiredAcceleration == dynamicResults.peakAcceleration;
	}
}
