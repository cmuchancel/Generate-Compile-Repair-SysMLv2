package 'InformationClassificationManagement' {
	private import ScalarValues::*;
	
	enum def InfoLevel {
		enum unclassified;
		enum confidential;
		enum secret;
	}
	
	part def InfoClass {
		attribute level : InfoLevel;
		attribute code : String;
		attribute color : String;
	}
	
	part unclassifiedInfo : InfoClass {
		attribute redefines level = InfoLevel::unclassified;
		attribute redefines code = "uncl";
		attribute redefines color = "green";
	}
	
	part confidentialInfo : InfoClass {
		attribute redefines level = InfoLevel::confidential;
		attribute redefines code = "conf";
		attribute redefines color = "yellow";
	}
	
	part secretInfo : InfoClass {
		attribute redefines level = InfoLevel::secret;
		attribute redefines code = "secr";
		attribute redefines color = "red";
	}
	
	enum def GradeLevel {
		enum A;
		enum B;
		enum C;
		enum D;
		enum F;
	}
	
	part def Grade {
		attribute level : GradeLevel;
		attribute points : Real;
	}
	
	part gradeA : Grade {
		attribute redefines level = GradeLevel::A;
		attribute redefines points = 4.0;
	}
	
	part gradeB : Grade {
		attribute redefines level = GradeLevel::B;
		attribute redefines points = 3.0;
	}
	
	part gradeC : Grade {
		attribute redefines level = GradeLevel::C;
		attribute redefines points = 2.0;
	}
	
	part gradeD : Grade {
		attribute redefines level = GradeLevel::D;
		attribute redefines points = 1.0;
	}
	
	part gradeF : Grade {
		attribute redefines level = GradeLevel::F;
		attribute redefines points = 0.0;
	}
}
