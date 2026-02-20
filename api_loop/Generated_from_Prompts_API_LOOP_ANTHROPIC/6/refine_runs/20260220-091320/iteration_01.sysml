package TrafficLightClassification {
    private import ScalarValues::*;

    enum def ClassificationLevel {
        enum UNCLASSIFIED;
        enum CONFIDENTIAL;
        enum SECRET;
    }

    enum def GradeLevel {
        enum A;
        enum B;
        enum C;
        enum D;
        enum F;
    }

    part def ClassificationEntry {
        attribute level : ClassificationLevel;
        attribute code : String;
        attribute color : String;
    }

    part def GradeEntry {
        attribute grade : GradeLevel;
        attribute points : Real;
    }

    part def ClassificationSystem {
        attribute unclassifiedCode : String = "uncl";
        attribute unclassifiedColor : String = "green";

        attribute confidentialCode : String = "conf";
        attribute confidentialColor : String = "yellow";

        attribute secretCode : String = "secr";
        attribute secretColor : String = "red";

        attribute gradeAPoints : Real = 4.0;
        attribute gradeBPoints : Real = 3.0;
        attribute gradeCPoints : Real = 2.0;
        attribute gradeDPoints : Real = 1.0;
        attribute gradeFPoints : Real = 0.0;

        part unclassifiedEntry : ClassificationEntry {
            attribute level : ClassificationLevel = ClassificationLevel::UNCLASSIFIED;
            attribute code : String = "uncl";
            attribute color : String = "green";
        }

        part confidentialEntry : ClassificationEntry {
            attribute level : ClassificationLevel = ClassificationLevel::CONFIDENTIAL;
            attribute code : String = "conf";
            attribute color : String = "yellow";
        }

        part secretEntry : ClassificationEntry {
            attribute level : ClassificationLevel = ClassificationLevel::SECRET;
            attribute code : String = "secr";
            attribute color : String = "red";
        }

        part gradeA : GradeEntry {
            attribute grade : GradeLevel = GradeLevel::A;
            attribute points : Real = 4.0;
        }

        part gradeB : GradeEntry {
            attribute grade : GradeLevel = GradeLevel::B;
            attribute points : Real = 3.0;
        }

        part gradeC : GradeEntry {
            attribute grade : GradeLevel = GradeLevel::C;
            attribute points : Real = 2.0;
        }

        part gradeD : GradeEntry {
            attribute grade : GradeLevel = GradeLevel::D;
            attribute points : Real = 1.0;
        }

        part gradeF : GradeEntry {
            attribute grade : GradeLevel = GradeLevel::F;
            attribute points : Real = 0.0;
        }
    }

    part system : ClassificationSystem;
}