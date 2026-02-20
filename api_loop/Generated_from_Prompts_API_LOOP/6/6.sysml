package ClassificationModel {
  private import ScalarValues::*;

  enum def InfoLevel {
    enum Unclassified;
    enum Confidential;
    enum Secret;
  }

  enum def GradeLevel {
    enum A;
    enum B;
    enum C;
    enum D;
    enum F;
  }

  part def InfoClassification {
    attribute level : InfoLevel;
    attribute code : String;
    attribute color : String;
  }

  part def GradeClassification {
    attribute level : GradeLevel;
    attribute points : Real;
  }

  part def ClassificationSystem {
    part uncl : InfoClassification {
      attribute level = InfoLevel::Unclassified;
      attribute code = "uncl";
      attribute color = "green";
    }
    part conf : InfoClassification {
      attribute level = InfoLevel::Confidential;
      attribute code = "conf";
      attribute color = "yellow";
    }
    part secr : InfoClassification {
      attribute level = InfoLevel::Secret;
      attribute code = "secr";
      attribute color = "red";
    }

    part gradeA : GradeClassification {
      attribute level = GradeLevel::A;
      attribute points = 4.0;
    }
    part gradeB : GradeClassification {
      attribute level = GradeLevel::B;
      attribute points = 3.0;
    }
    part gradeC : GradeClassification {
      attribute level = GradeLevel::C;
      attribute points = 2.0;
    }
    part gradeD : GradeClassification {
      attribute level = GradeLevel::D;
      attribute points = 1.0;
    }
    part gradeF : GradeClassification {
      attribute level = GradeLevel::F;
      attribute points = 0.0;
    }
  }
}