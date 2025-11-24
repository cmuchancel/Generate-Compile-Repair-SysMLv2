package 'EngineeringMaterialsDatabase' {
	private import ScalarValues::*;
	private import ISQ::*;
	private import SI::*;
	abstract part def Material {
		attribute name : String;
		attribute atomicMass : Real;
		attribute description : String;
	}
	part def Metal :> Material;
	part def AlloyComponent {
		ref part constituent : Material;
		attribute massFraction : Real;
	}
	part def Alloy :> Material {
		attribute tensileStrength : ISQ::PressureValue;
		part components : AlloyComponent[2..*];
	}
	calc def ConvertTensileStrength {
		in baseValue : ISQ::PressureValue;
		return convertedValue : ISQ::PressureValue;
	}
	part materialRepository {
		part iron : Metal {
			attribute name = "Iron";
			attribute atomicMass = 55.845;
			attribute description = "Base metal for steel alloys.";
		}
		part carbon : Metal {
			attribute name = "Carbon";
			attribute atomicMass = 12.011;
			attribute description = "Light element improving hardness.";
		}
		part manganese : Metal {
			attribute name = "Manganese";
			attribute atomicMass = 54.938;
			attribute description = "Alloying metal that adds strength.";
		}
	}
	part alloyRepository {
		part steel_980 : Alloy {
			attribute name = "Steel_980";
			attribute atomicMass = 55.0;
			attribute description = "High-strength automotive steel.";
			attribute tensileStrength = 9.8E8 [SI::Pa];
			part ironFraction : AlloyComponent {
				ref part constituent = materialRepository::iron;
				attribute massFraction = 0.965;
			}
			part carbonFraction : AlloyComponent {
				ref part constituent = materialRepository::carbon;
				attribute massFraction = 0.02;
			}
			part manganeseFraction : AlloyComponent {
				ref part constituent = materialRepository::manganese;
				attribute massFraction = 0.015;
			}
			calc convertSteelStrength : ConvertTensileStrength {
				in baseValue = steel_980.tensileStrength;
				return convertedValue = steel_980.tensileStrength;
			}
		}
	}
	requirement def 'Unit Conversion Requirement' {
		text = "Manage tensile strength using standardized dimensions and provide conversions to engineering units such as newtons per square millimeter.";
	}
	requirement def 'Material Data Management Requirement' {
		text = "Provide intuitive, standardized, and flexible data entry, querying, and comparison to support material selection, evaluation, and application.";
	}
}
