package 'VehicleSafetySecurityCompliance' {
	private import ScalarValues::*;
	package AnnotationDefinitions {
		metadata def Safety {
			attribute isMandatory : Boolean;
		}
		metadata def Security {
			attribute isRequired : Boolean;
		}
	}
	private import AnnotationDefinitions::*;
	part def Vehicle {
		part interior {
			part seatBelt[2] {@Safety{isMandatory = true;}}
			part airBag {@Safety{isMandatory = false;}}
			part alarm {@Security{isRequired = true;}}
		}
		part body {
			part bumper {@Safety{isMandatory = true;}}
			part keylessEntry {@Security{isRequired = true;}}
		}
		part chassis {
			part wheel[2];
			part antiLockBrakingSystem[2] {@Safety{isMandatory = false;}}
		}
	}
	part vehicle : Vehicle;
	package 'Safety Components' {
		private import vehicle::**;
		filter @Safety;
	}
	package 'Mandatory Safety Components' {
		private import vehicle::**;
		filter @Safety and Safety::isMandatory;
	}
	package 'Optional Safety Components' {
		private import vehicle::**;
		filter @Safety and not Safety::isMandatory;
	}
	package 'Security Components' {
		private import vehicle::**;
		filter @Security;
	}
	package 'Required Security Components' {
		private import vehicle::**;
		filter @Security and Security::isRequired;
	}
}
