package 'VehicleMassVerification076' {
	private import ScalarValues::*;
	
	part def Vehicle {
		attribute measuredMass : Real;
	}
	
	requirement def VehicleMassRequirement {
		subject vehicle : Vehicle;
		attribute maxMass : Real = 2500.0;
		require constraint { vehicle.measuredMass <= maxMass }
	}
	
	verification def VehicleMassInspection {
		subject inspectedVehicle : Vehicle;
		requirement vehicleMassRequirement : VehicleMassRequirement {
			subject vehicle = inspectedVehicle;
		}
		objective verifyVehicleMass {
			verify vehicleMassRequirement;
		}
		part weighingEquipment {
			action measureMass {
				out measuredMass : Real;
			}
		}
		action processMeasurement {
			in measuredMass : Real = weighingEquipment.measureMass.measuredMass;
			out processedMass : Real = measuredMass;
		}
		action evaluateMeasurement {
			in processedMass : Real = processMeasurement.processedMass;
			out verdict : Boolean = processedMass <= VehicleMassRequirement::maxMass;
		}
		return verificationVerdict : Boolean = evaluateMeasurement.verdict;
	}
	
	part vehicleInspectionContext {
		part vehicleUnderTest : Vehicle;
		requirement vehicleMassRequirement : VehicleMassRequirement {
			subject vehicle = vehicleInspectionContext.vehicleUnderTest;
		}
		verification inspection : VehicleMassInspection {
			subject inspectedVehicle = vehicleInspectionContext.vehicleUnderTest;
		}
	}
}
