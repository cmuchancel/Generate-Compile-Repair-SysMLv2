package 'FuelEconomyCompliance' {
	private import ScalarValues::*;
	enum def DrivingScenario {
		enum Urban;
		enum Highway;
	}
	part def Vehicle {
		attribute powertrainEfficiency : Real;
		attribute baselineLoad : Real;
	}
	part def FuelEconomyTest {
		part vehicle : Vehicle;
		attribute scenario : DrivingScenario;
		attribute additionalLoad : Real;
		attribute milesPerGallon : Real;
	}
	constraint def FuelEconomyRequirement {
		in minimumFuelEconomy : Real;
		in measuredFuelEconomy : Real;
		assert constraint { measuredFuelEconomy >= minimumFuelEconomy }
	}
	requirement def 'Urban Fuel Economy Requirement' {
		text = "Urban driving fuel economy shall be at least 25 miles per US gallon with 1,000 pound load.";
	}
	requirement def 'Highway Fuel Economy Requirement' {
		text = "Highway driving fuel economy shall be at least 30 miles per US gallon with 1,000 pound load.";
	}
	part complianceAssessment {
		part vehicle : Vehicle {
			attribute powertrainEfficiency = 0.31;
			attribute baselineLoad = 3500;
		}
		part urbanTest : FuelEconomyTest {
			part vehicle = complianceAssessment::vehicle;
			attribute scenario = DrivingScenario::Urban;
			attribute additionalLoad = 1000;
			attribute milesPerGallon = 26;
		}
		part highwayTest : FuelEconomyTest {
			part vehicle = complianceAssessment::vehicle;
			attribute scenario = DrivingScenario::Highway;
			attribute additionalLoad = 1000;
			attribute milesPerGallon = 32;
		}
		part urbanCompliance : FuelEconomyRequirement {
			in minimumFuelEconomy = 25;
			in measuredFuelEconomy = urbanTest.milesPerGallon;
		}
		part highwayCompliance : FuelEconomyRequirement {
			in minimumFuelEconomy = 30;
			in measuredFuelEconomy = highwayTest.milesPerGallon;
		}
		requirement urbanRequirement : 'Urban Fuel Economy Requirement';
		requirement highwayRequirement : 'Highway Fuel Economy Requirement';
	}
}
