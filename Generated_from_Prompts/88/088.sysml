package 'VehicleSafetyCompliance' {
	private import ScalarValues::*;
	private import AnnotationDefinitions::*;
	package AnnotationDefinitions {
		metadata def Safety {
			attribute isMandatory : Boolean;
		}
	}
	part def SeatBelt;
	part def AirBag;
	part def Bumper;
	part def KeylessEntrySystem;
	part def Tire;
	part def AntiLockBrakingSystem;
	package VehicleStructure {
		part vehicle {
			part interior {
				part seatBelt : SeatBelt[2] {@Safety{isMandatory = true;}}
				part driverAirBag : AirBag {@Safety{isMandatory = false;}}
			}
			part body {
				part bumper : Bumper {@Safety{isMandatory = true;}}
				part keylessEntry : KeylessEntrySystem {@Safety{isMandatory = false;}}
			}
			part chassis {
				part tires : Tire[4] {@Safety{isMandatory = true;}}
				part abs : AntiLockBrakingSystem {@Safety{isMandatory = false;}}
			}
		}
	}
	package 'Safety Components' {
		private import VehicleStructure::vehicle::**;
		filter @Safety;
	}
	package 'Mandatory Safety Components' {
		private import 'Safety Components'::*;
		filter Safety::isMandatory;
	}
	package 'Optional Safety Components' {
		private import 'Safety Components'::*;
		filter not Safety::isMandatory;
	}
}
