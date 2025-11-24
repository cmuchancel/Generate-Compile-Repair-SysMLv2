package 'AutomotiveSafetyArchitecture' {
	private import ScalarValues::*;
	private import Views::*;
	package AnnotationDefinitions {
		metadata def SafetyMarker {
			attribute isMandatory : Boolean;
		}
	}
	private import AnnotationDefinitions::*;
	port def SystemInterface {
		attribute description : String;
	}
	part def ComponentElement {
		attribute functionalResponsibility : String;
		port integration : SystemInterface;
	}
	part def SafetyComponent :> ComponentElement;
	part def VehicleSystem :> ComponentElement {
		part safetySubsystem : ComponentElement {
			attribute redefines functionalResponsibility = "Safety governance";
			port integration : SystemInterface { attribute description = "Safety coordination"; }
			part seatBelts : SafetyComponent {
				attribute functionalResponsibility = "Occupant restraint";
				port integration : SystemInterface { attribute description = "Seat belt anchors"; }
				@SafetyMarker{isMandatory = true;}
			}
			part bumpers : SafetyComponent {
				attribute functionalResponsibility = "Impact absorption";
				port integration : SystemInterface { attribute description = "Crash impact channel"; }
				@SafetyMarker{isMandatory = true;}
			}
			part driverAirbag : SafetyComponent {
				attribute functionalResponsibility = "Driver frontal protection";
				port integration : SystemInterface { attribute description = "Airbag deployment interface"; }
				@SafetyMarker{isMandatory = false;}
			}
			part antiLockBrakingSystem : SafetyComponent {
				attribute functionalResponsibility = "Wheel slip control";
				port integration : SystemInterface { attribute description = "Brake modulation channel"; }
				@SafetyMarker{isMandatory = false;}
			}
		}
		part functionalSubsystem : ComponentElement {
			attribute redefines functionalResponsibility = "Core vehicle functionality";
			port integration : SystemInterface { attribute description = "Functional coordination"; }
			part powertrainController : ComponentElement {
				attribute functionalResponsibility = "Propulsion management";
				port integration : SystemInterface { attribute description = "Powertrain bus"; }
			}
			part infotainmentSystem : ComponentElement {
				attribute functionalResponsibility = "User information services";
				port integration : SystemInterface { attribute description = "Cabin user interface"; }
			}
			part climateController : ComponentElement {
				attribute functionalResponsibility = "Cabin thermal comfort";
				port integration : SystemInterface { attribute description = "HVAC routing"; }
			}
		}
		part environmentInterface : ComponentElement {
			attribute redefines functionalResponsibility = "Environmental awareness";
			port integration : SystemInterface { attribute description = "External sensing"; }
		}
	}
	part automotiveSystem : VehicleSystem {
		attribute redefines functionalResponsibility = "Automotive root component";
		port integration : SystemInterface { attribute description = "System coordination hub"; }
	}
	package 'Component Hierarchy' {
		private import automotiveSystem::**;
	}
	package 'Safety Components' {
		private import automotiveSystem::**;
		filter @SafetyMarker;
	}
	package 'Mandatory Safety Components' {
		private import automotiveSystem::**;
		filter @SafetyMarker and SafetyMarker::isMandatory;
	}
	package 'Optional Safety Components' {
		private import automotiveSystem::**;
		filter @SafetyMarker and not SafetyMarker::isMandatory;
	}
	package 'Non-Safety Components' {
		private import automotiveSystem::**;
		filter not @SafetyMarker;
	}
	view 'Component Tree' {
		expose 'Component Hierarchy'::*;
		render asTreeDiagram;
	}
	view 'Component Table' {
		expose 'Component Hierarchy'::*;
		render asElementTable {
			view redefines columnView[1] {
				ref rendering = Views::asTextualNotation;
			}
		}
	}
	view 'Safety Component Tree' {
		expose 'Safety Components'::*;
		render asTreeDiagram;
	}
	view 'Safety Component Table' {
		expose 'Safety Components'::*;
		render asElementTable {
			view redefines columnView[1] {
				ref rendering = Views::asTextualNotation;
			}
		}
	}
	view 'Non-Safety Component Tree' {
		expose 'Non-Safety Components'::*;
		render asTreeDiagram;
	}
	view 'Non-Safety Component Table' {
		expose 'Non-Safety Components'::*;
		render asElementTable {
			view redefines columnView[1] {
				ref rendering = Views::asTextualNotation;
			}
		}
	}
	requirement def 'Stakeholder View Requirement' {
		text = "Provide tree and tabular views covering component hierarchy, safety features, and non-safety components for use by system engineers and independent V&V personnel.";
	}
}
