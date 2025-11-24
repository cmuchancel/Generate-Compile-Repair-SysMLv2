package 'SecurityClassificationManagement' {
	private import ScalarValues::*;
	package AnnotationDefinitions {
		enum def ClassificationLevel {
			enum Unclassified;
			enum Confidential;
			enum Secret;
		}
		metadata def ClassificationTag {
			attribute level : ClassificationLevel;
			attribute isConfidential : Boolean;
		}
	}
	private import AnnotationDefinitions::*;
	part def Component;
	part componentRepository {
		part dataBus : Component {@ClassificationTag{level = ClassificationLevel::Unclassified; isConfidential = false;}}
		part missionComputer : Component {@ClassificationTag{level = ClassificationLevel::Confidential; isConfidential = true;}}
		part threatAnalysisModule : Component {@ClassificationTag{level = ClassificationLevel::Secret; isConfidential = false;}}
	}
	package 'Confidential Components' {
		private import componentRepository::**;
		filter @ClassificationTag and ClassificationTag::isConfidential;
	}
}
