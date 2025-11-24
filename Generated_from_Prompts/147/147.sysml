package 'AlloyMaterialManagement' {
	private import ScalarValues::*;
	private import ISQ::*;
	private import SI::*;
	abstract part def Substance {
		attribute designation : String;
		attribute atomicMass : Real;
	}
	part def Metal :> Substance;
	part def AlloyComponent {
		ref part material : Substance;
		attribute massFraction : Real;
	}
	part def Alloy {
		attribute designation : String;
		attribute tensileStrength : ISQ::PressureValue;
		part components : AlloyComponent[2..*];
	}
	part materialCatalog {
		part iron : Metal {
			attribute designation = "Iron";
			attribute atomicMass = 55.845;
		}
		part carbon : Metal {
			attribute designation = "Carbon";
			attribute atomicMass = 12.011;
		}
		part manganese : Metal {
			attribute designation = "Manganese";
			attribute atomicMass = 54.938;
		}
	}
	part alloyCatalog {
		part steel980 : Alloy {
			attribute designation = "Steel 980";
			attribute tensileStrength = 9.8E8 [SI::Pa];
			part ironFraction : AlloyComponent {
				ref part material = materialCatalog::iron;
				attribute massFraction = 0.965;
			}
			part carbonFraction : AlloyComponent {
				ref part material = materialCatalog::carbon;
				attribute massFraction = 0.02;
			}
			part manganeseFraction : AlloyComponent {
				ref part material = materialCatalog::manganese;
				attribute massFraction = 0.015;
			}
		}
	}
	requirement def 'Derived Unit Requirement' {
		text = "Tensile strength of every alloy shall be expressed using derived SI units, including newtons per square millimeter.";
	}
	requirement def 'Metallurgy Data Management Requirement' {
		text = "Provide editable and extensible alloy composition and property data to support scientific research, engineering design, and quality control.";
	}
}
