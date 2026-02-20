package 'InformationClassification' {
	private import ScalarValues::*;

	part def InformationLevel {
		attribute name : String;
		attribute code : String;
		attribute color : SignalColorKind;
	}

	part def UnclassifiedInformation specializes InformationLevel {
		attribute name redefines InformationLevel::name : String = "Unclassified";
		attribute code redefines InformationLevel::code : String = "uncl";
		attribute color redefines InformationLevel::color : SignalColorKind = SignalColorKind::green;
	}

	part def ConfidentialInformation specializes InformationLevel {
		attribute name redefines InformationLevel::name : String = "Confidential";
		attribute code redefines InformationLevel::code : String = "conf";
		attribute color redefines InformationLevel::color : SignalColorKind = SignalColorKind::yellow;
	}

	part def SecretInformation specializes InformationLevel {
		attribute name redefines InformationLevel::name : String = "Secret";
		attribute code redefines InformationLevel::code : String = "secr";
		attribute color redefines InformationLevel::color : SignalColorKind = SignalColorKind::red;
	}

	enum def SignalColorKind {
		enum green;
		enum yellow;
		enum red;
	}

	part def GradeLevel {
		attribute letter : GradeLetterKind;
		attribute points : Real;
	}

	part def GradeA specializes GradeLevel {
		attribute letter redefines GradeLevel::letter : GradeLetterKind = GradeLetterKind::A;
		attribute points redefines GradeLevel::points : Real = 4.0;
	}

	part def GradeB specializes GradeLevel {
		attribute letter redefines GradeLevel::letter : GradeLetterKind = GradeLetterKind::B;
		attribute points redefines GradeLevel::points : Real = 3.0;
	}

	part def GradeC specializes GradeLevel {
		attribute letter redefines GradeLevel::letter : GradeLetterKind = GradeLetterKind::C;
		attribute points redefines GradeLevel::points : Real = 2.0;
	}

	part def GradeD specializes GradeLevel {
		attribute letter redefines GradeLevel::letter : GradeLetterKind = GradeLetterKind::D;
		attribute points redefines GradeLevel::points : Real = 1.0;
	}

	part def GradeF specializes GradeLevel {
		attribute letter redefines GradeLevel::letter : GradeLetterKind = GradeLetterKind::F;
		attribute points redefines GradeLevel::points : Real = 0.0;
	}

	enum def GradeLetterKind {
		enum A;
		enum B;
		enum C;
		enum D;
		enum F;
	}
}
