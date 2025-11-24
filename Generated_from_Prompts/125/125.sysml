package 'VehicleSafetySecurityView' {
	private import ScalarValues::*;
	private import Views::*;
	package AnnotationDefinitions {
		metadata def Safety {
			attribute isMandatory : Boolean;
		}
		metadata def Security;
	}
	private import AnnotationDefinitions::*;
	part def Vehicle {
		part interior {
			part seatBelt[2] {@Safety{isMandatory = true;}}
			part airBag {@Safety{isMandatory = false;}}
			part alarm {@Security;}
		}
		part body {
			part bumper {@Safety{isMandatory = true;}}
			part keylessEntry {@Security;}
		}
		part chassis {
			part antiLockBrakingSystem {@Safety{isMandatory = false;}}
		}
	}
	part vehicle : Vehicle;
	package 'Safety Features' {
		private import vehicle::**;
		filter @Safety;
	}
	package 'Mandatory Safety Features' {
		private import vehicle::**;
		filter @Safety and Safety::isMandatory;
	}
	package 'Optional Safety Features' {
		private import vehicle::**;
		filter @Safety and not Safety::isMandatory;
	}
	package 'Security Features' {
		private import vehicle::**;
		filter @Security;
	}
	view 'Safety Feature Tree' {
		expose 'Safety Features'::*;
		render asTreeDiagram;
	}
	view 'Safety Feature Table' {
		expose 'Safety Features'::*;
		render asElementTable {
			view redefines columnView[1] {
				ref rendering = Views::asTextualNotation;
			}
		}
	}
	view 'Security Feature Table' {
		expose 'Security Features'::*;
		render asElementTable {
			view redefines columnView[1] {
				ref rendering = Views::asTextualNotation;
			}
		}
	}
}
