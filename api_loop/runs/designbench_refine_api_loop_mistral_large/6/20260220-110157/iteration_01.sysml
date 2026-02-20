package InformationClassification {

    private import ScalarValues::*;

    enum def ClassificationLevel {
        enum uncl;
        enum conf;
        enum secr;
    }

    enum def ClassificationColor {
        enum green;
        enum yellow;
        enum red;
    }

    enum def GradeLevel {
        enum A;
        enum B;
        enum C;
        enum D;
        enum F;
    }

    part def Classification {
        attribute level : ClassificationLevel;
        attribute code : String;
        attribute color : ClassificationColor;
    }

    part unclassified : Classification {
        attribute level = ClassificationLevel::uncl;
        attribute code = "uncl";
        attribute color = ClassificationColor::green;
    }

    part confidential : Classification {
        attribute level = ClassificationLevel::conf;
        attribute code = "conf";
        attribute color = ClassificationColor::yellow;
    }

    part secret : Classification {
        attribute level = ClassificationLevel::secr;
        attribute code = "secr";
        attribute color = ClassificationColor::red;
    }

    part def Grade {
        attribute grade : GradeLevel;
        attribute points : Real;
    }

    part gradeA : Grade {
        attribute grade = GradeLevel::A;
        attribute points = 4.0;
    }

    part gradeB : Grade {
        attribute grade = GradeLevel::B;
        attribute points = 3.0;
    }

    part gradeC : Grade {
        attribute grade = GradeLevel::C;
        attribute points = 2.0;
    }

    part gradeD : Grade {
        attribute grade = GradeLevel::D;
        attribute points = 1.0;
    }

    part gradeF : Grade {
        attribute grade = GradeLevel::F;
        attribute points = 0.0;
    }

    part def Information {
        attribute classification : Classification;
        attribute grade : Grade;
    }

}