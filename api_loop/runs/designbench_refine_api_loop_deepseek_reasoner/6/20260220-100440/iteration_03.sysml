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
        attribute levelValue = InformationLevel::unclassified;
        attribute codeValue = "uncl";
        attribute colorValue = "green";
    }

    part ConfidentialInfo : TrafficLightInfo {
        attribute levelValue = InformationLevel::confidential;
        attribute codeValue = "conf";
        attribute colorValue = "yellow";
    }

    part SecretInfo : TrafficLightInfo {
        attribute levelValue = InformationLevel::secret;
        attribute codeValue = "secr";
        attribute colorValue = "red";
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
        attribute gradeValue = Grade::A;
        attribute pointsValue = 4.0;
    }

    part GradeB : GradeInfo {
        attribute gradeValue = Grade::B;
        attribute pointsValue = 3.0;
    }

    part GradeC : GradeInfo {
        attribute gradeValue = Grade::C;
        attribute pointsValue = 2.0;
    }

    part GradeD : GradeInfo {
        attribute gradeValue = Grade::D;
        attribute pointsValue = 1.0;
    }

    part GradeF : GradeInfo {
        attribute gradeValue = Grade::F;
        attribute pointsValue = 0.0;
    }

    part def ClassifiedItem {
        attribute infoClassification : TrafficLightInfo;
        attribute gradeClassification : GradeInfo;
    }

    part def ManagementSystem {
        part items : ClassifiedItem[0..*];
    }
}