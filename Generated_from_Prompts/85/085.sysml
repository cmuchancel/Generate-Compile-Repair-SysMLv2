package 'VehicleFeatureAnnotationSystem' {
	package AnnotationDefinitions {
		metadata def Safety;
		metadata def Security;
	}
	package VehicleStructure {
		private import AnnotationDefinitions::*;
		part vehicle {
			part interior {
				part seatBelt[2] {@Safety;}
				part frontSeat[2];
				part driverAirBag {@Safety;}
				part alarmSystem {@Security;}
			}
			part bodyAssembly {
				part bodyShell;
				part bumper {@Safety;}
				part keylessEntry {@Security;}
			}
		}
	}
	package 'Safety Features' {
		private import AnnotationDefinitions::*;
		private import VehicleStructure::vehicle::**;
		filter @Safety;
	}
	package 'Security Features' {
		private import AnnotationDefinitions::*;
		private import VehicleStructure::vehicle::**;
		filter @Security;
	}
}
