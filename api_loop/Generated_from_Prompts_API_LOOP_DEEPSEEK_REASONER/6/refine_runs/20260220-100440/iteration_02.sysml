package InformationClassification {
    private import ScalarValues::*;

    enum def InformationLevel {
        enum unclassified;
        enum confidential;
        enum secret;
    }

    part def TrafficLightInfo {
        attribute level : InformationLevel;
        attribute code : string;
        attribute color : string;
    }

    part UnclassifiedInfo : TrafficLightInfo {
        attribute level = InformationLevel::unclassified;
        attribute code = "uncl";
        attribute color = "green";
    }

    part ConfidentialInfo : TrafficLightInfo {
        attribute level = InformationLevel::confidential;
        attribute code = "conf";
        attribute color = "yellow";
    }

    part SecretInfo : TrafficLightInfo {
        attribute level = InformationLevel::secret;
        attribute code = "secr";
        attribute color = "red";
    }

    enum def Grade {
        enum A;
        enum B;
        enum C;
        enum D;
        enum F;
    }

    part def GradeInfo {
        attribute grade : Grade;
        attribute points : real;
    }

    part GradeA : GradeInfo {
        attribute grade = Grade::A;
        attribute points = 4.0;
    }

    part GradeB : GradeInfo {
        attribute grade = Grade::B;
        attribute points = 3.0;
    }

    part GradeC : GradeInfo {
        attribute grade = Grade::C;
        attribute points = 2.0;
    }

    part GradeD : GradeInfo {
        attribute grade = Grade::D;
        attribute points = 1.0;
    }

    part GradeF : GradeInfo {
        attribute grade = Grade::F;
        attribute points = 0.0;
    }

    part def ClassifiedItem {
        attribute infoClassification : TrafficLightInfo;
        attribute gradeClassification : GradeInfo;
    }

    part def ManagementSystem {
        part items : ClassifiedItem[0..*];
    }
}