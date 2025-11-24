library package 'AutomotiveGlossary' {
	private import ScalarValues::*;
	public import ISQ::TorqueValue;
	doc
	/*
	 * This glossary provides canonical system elements for the automotive
	 * domain, creating a shared vocabulary for downstream models.
	 */
	part def '汽车' {
		doc
		/*
		 * Core automotive system element representing a complete vehicle.
		 * It provides the anchor for integrating functional and safety
		 * subsystems across product lines.
		 */
	}
	public alias '汽车 (Car)' for '汽车' {
		doc
		/*
		 * Public alias that pairs the Chinese term with the English
		 * translation to simplify communication among mixed-language teams.
		 */
	}
	public alias Car for '汽车';
	public alias Automobile for '汽车';
	public alias '扭矩 (Torque)' for TorqueValue {
		doc
		/*
		 * TorqueValue from the ISQ library is aliased to the bilingual
		 * label used by calibration and controls engineers.
		 */
	}
	metadata def DomainNote {
		attribute meaning : String;
		attribute usageContext : String;
	}
	part conceptLibrary {
		part vehicle : '汽车' {@DomainNote{meaning = "Complete automobile platform"; usageContext = "All vehicle-level discussions";}}
		attribute ratedTorque : TorqueValue {@DomainNote{meaning = "Nominal drivetrain torque"; usageContext = "Powertrain sizing";}}
	}
}
