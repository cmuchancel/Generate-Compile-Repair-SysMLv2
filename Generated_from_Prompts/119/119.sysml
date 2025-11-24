package 'VehicleMassComplianceVerification' {
	private import ScalarValues::*;
	part def Vehicle {
		attribute identifier : String;
		attribute actualMassKilograms : Real;
		attribute complianceStatus : String;
	}
	part def TestOperator {
		attribute name : String;
	}
	part def WeighingEquipment {
		attribute manufacturer : String;
	}
	constraint def MassLimitRequirement {
		in measuredMass : Real;
		in limit : Real;
		measuredMass <= limit
	}
	action def PerformWeighing {
		in operator : TestOperator;
		in equipment : WeighingEquipment;
		in vehicle : Vehicle;
		out measuredMass : Real;
	}
	action def DetermineCompliance {
		in measuredMass : Real;
		in limit : Real;
		out isCompliant : Boolean;
	}
	part def VehicleTestCase {
		attribute limitKilograms : Real;
		part vehicle : Vehicle;
		part operator : TestOperator;
		part equipment : WeighingEquipment;
		assert constraint complianceCheck : MassLimitRequirement {
			in measuredMass = vehicle.actualMassKilograms;
			in limit = limitKilograms;
		}
	}
	part def ComplianceTestCampaign {
		attribute limitKilograms = 2500.0;
		part testCases : VehicleTestCase[*];
	}
	part campaign : ComplianceTestCampaign {
		part testCases[2] : VehicleTestCase redefines testCases {
			part vehicleA : VehicleTestCase redefines testCases[1] {
				attribute limitKilograms = campaign.limitKilograms;
				part vehicle : Vehicle redefines vehicle {
					attribute identifier = "Vehicle-A";
					attribute actualMassKilograms = 2450.0;
					attribute complianceStatus = "Pending";
				}
				part operator : TestOperator redefines operator {
					attribute name = "Operator-1";
				}
				part equipment : WeighingEquipment redefines equipment {
					attribute manufacturer = "CertifiedScaleCo";
				}
			}
			part vehicleB : VehicleTestCase redefines testCases[2] {
				attribute limitKilograms = campaign.limitKilograms;
				part vehicle : Vehicle redefines vehicle {
					attribute identifier = "Vehicle-B";
					attribute actualMassKilograms = 2600.0;
					attribute complianceStatus = "Pending";
				}
				part operator : TestOperator redefines operator {
					attribute name = "Operator-2";
				}
				part equipment : WeighingEquipment redefines equipment {
					attribute manufacturer = "CertifiedScaleCo";
				}
			}
		}
	}
	action weighingVehicleA : PerformWeighing {
		in operator = campaign.testCases.vehicleA.operator;
		in equipment = campaign.testCases.vehicleA.equipment;
		in vehicle = campaign.testCases.vehicleA.vehicle;
		out measuredMass = campaign.testCases.vehicleA.vehicle.actualMassKilograms;
	}
	action weighingVehicleB : PerformWeighing {
		in operator = campaign.testCases.vehicleB.operator;
		in equipment = campaign.testCases.vehicleB.equipment;
		in vehicle = campaign.testCases.vehicleB.vehicle;
		out measuredMass = campaign.testCases.vehicleB.vehicle.actualMassKilograms;
	}
	action complianceVehicleA : DetermineCompliance {
		in measuredMass = campaign.testCases.vehicleA.vehicle.actualMassKilograms;
		in limit = campaign.limitKilograms;
		out isCompliant = true;
	}
	action complianceVehicleB : DetermineCompliance {
		in measuredMass = campaign.testCases.vehicleB.vehicle.actualMassKilograms;
		in limit = campaign.limitKilograms;
		out isCompliant = false;
	}
	requirement def 'MassComplianceRequirement' {
		text = "Verify that each measured vehicle mass is less than or equal to the 2,500 kilogram limit; otherwise mark the vehicle as non-compliant.";
	}
	requirement def 'SequentialTestingRequirement' {
		text = "Support automated, sequential weighing of multiple vehicles using operators and certified equipment, recording each result for compliance management.";
	}
}
