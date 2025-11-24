package 'VehicleDynamicsAnalysis072' {
	private import ScalarValues::*;
	
	part def VehicleParameters {
		attribute curbWeight : Real;
		attribute payload : Real;
		attribute wheelDiameter : Real;
		attribute transmissionEfficiency : Real;
		attribute urbanFuelEconomyMpg : Real;
		attribute highwayFuelEconomyMpg : Real;
	}
	
	part def DrivePoint {
		attribute timestamp : Real;
		attribute position : Real;
		attribute speed : Real;
	}
	
	part def DrivingCycle {
		part point : DrivePoint[*] ordered;
	}
	
	part def DriveSegment {
		attribute initialSpeed : Real;
		attribute finalSpeed : Real;
		attribute timeInterval : Real;
		attribute distanceMiles : Real;
		attribute fuelGallonsUsed : Real;
	}
	
	calc def ComputeAcceleration {
		in initialSpeed : Real;
		in finalSpeed : Real;
		in timeInterval : Real;
		return acceleration : Real = (finalSpeed - initialSpeed) / timeInterval;
	}
	
	calc def ComputeRequiredPower {
		in vehicleMass : Real;
		in acceleration : Real;
		in speed : Real;
		in transmissionEfficiency : Real;
		return power : Real = (vehicleMass * acceleration * speed) / transmissionEfficiency;
	}
	
	calc def ComputeFuelEconomyMpg {
		in distanceMiles : Real;
		in fuelGallons : Real;
		return mpg : Real = distanceMiles / fuelGallons;
	}
	
	part performanceAnalysis {
		part vehicle : VehicleParameters;
		part cycle : DrivingCycle;
		part segment : DriveSegment;
		attribute targetFuelEconomyMpg : Real = 30.0;
		attribute vehicleMass : Real = vehicle.curbWeight + vehicle.payload;
		calc accelerationCalc : ComputeAcceleration {
			in initialSpeed = segment.initialSpeed;
			in finalSpeed = segment.finalSpeed;
			in timeInterval = segment.timeInterval;
			return acceleration;
		}
		calc requiredPowerCalc : ComputeRequiredPower {
			in vehicleMass = performanceAnalysis.vehicleMass;
			in acceleration = accelerationCalc.acceleration;
			in speed = segment.finalSpeed;
			in transmissionEfficiency = vehicle.transmissionEfficiency;
			return power;
		}
		calc fuelEconomyCalc : ComputeFuelEconomyMpg {
			in distanceMiles = segment.distanceMiles;
			in fuelGallons = segment.fuelGallonsUsed;
			return mpg;
		}
		attribute compliantFuelEconomy : Boolean = fuelEconomyCalc.mpg >= targetFuelEconomyMpg;
	}
}
