package 'HeadlampTiltValidation' {
	private import ScalarValues::*;
	private import SI::*;
	private import MeasurementReferences::*;
	package MathematicalReferences {
		attribute piConstant : Real = 3.141592653589793;
		attribute eConstant : Real = 2.718281828459045;
	}
	private import MathematicalReferences::*;
	part def HeadlampTiltAdjustmentKnob {
		attribute tiltAngleDegrees : Real;
		attribute tiltAngleRadians : Real;
		attribute toleranceFactor : Real = 1.0 / eConstant;
		attribute radianConversionFactor : Real = piConstant / 180.0;
		attribute siPlaneAngleUnit : DimensionOneValue = 1 [one];
	}
	part headlampValidationSuite {
		part adjustmentKnob : HeadlampTiltAdjustmentKnob;
		requirement tiltRangeRequirement : 'Beam Tilt Range Requirement' {
			subject knob = adjustmentKnob;
		}
		requirement precisionRequirement : 'SI Precision Requirement' {
			subject knob = adjustmentKnob;
		}
	}
	requirement def 'Beam Tilt Range Requirement' {
		subject knob : HeadlampTiltAdjustmentKnob;
		text = "Headlamp beam tilt angles, measured in degrees, must remain between 50 and 80 degrees to comply with safety and regulatory standards.";
		require constraint { knob.tiltAngleDegrees >= 50.0 and knob.tiltAngleDegrees <= 80.0 }
	}
	requirement def 'SI Precision Requirement' {
		subject knob : HeadlampTiltAdjustmentKnob;
		text = "The verification system shall adopt SI-based calculations and the standard real number constants pi and Euler's number e to ensure scientific, reliable control of the adjustment knob.";
	}
}
