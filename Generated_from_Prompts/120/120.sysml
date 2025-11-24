package 'VehicleQualityAnalysis' {
	private import ScalarValues::*;
	part def Engine {
		attribute mass : Real;
	}
	part def Transmission {
		attribute mass : Real;
	}
	part def AxleAssembly {
		attribute mass : Real;
	}
	part def Vehicle {
		attribute totalMass : Real;
		part engine : Engine;
		part transmission : Transmission;
		part frontAxleAssembly : AxleAssembly;
		part rearAxleAssembly : AxleAssembly;
	}
	constraint def TotalMassConstraint {
		in engineMass : Real;
		in transmissionMass : Real;
		in frontAxleMass : Real;
		in rearAxleMass : Real;
		in totalMass : Real;
		assert constraint { totalMass = engineMass + transmissionMass + frontAxleMass + rearAxleMass; }
	}
	requirement def 'Quality Analysis Requirement' {
		text = "Aggregate vehicle component masses and track quality metrics.";
	}
	use case def 'Vehicle Quality Analysis Use Case';
	use case def 'Generate Analysis Plan';
	part qualityAnalysisSystem {
		part vehicle : Vehicle;
		part totalMassCheck : TotalMassConstraint {
			in engineMass = vehicle.engine.mass;
			in transmissionMass = vehicle.transmission.mass;
			in frontAxleMass = vehicle.frontAxleAssembly.mass;
			in rearAxleMass = vehicle.rearAxleAssembly.mass;
			in totalMass = vehicle.totalMass;
		}
		requirement qualityRequirement : 'Quality Analysis Requirement';
		use case analysisUseCase : 'Vehicle Quality Analysis Use Case';
		use case planUseCase : 'Generate Analysis Plan';
	}
}
