package SecurityClassification {

    private import ScalarValues::*;

    enum def SecurityLevel {
        enum Unclassified;
        enum Confidential;
        enum Secret;
    }

    part def Component {
        attribute classification : SecurityLevel;
    }

    part confidentialComponent : Component {
        attribute classification : SecurityLevel = SecurityLevel::Confidential;
    }

    part unclassifiedComponent : Component {
        attribute classification : SecurityLevel = SecurityLevel::Unclassified;
    }

    part secretComponent : Component {
        attribute classification : SecurityLevel = SecurityLevel::Secret;
    }

    part def System {
        part components : Component[0..*];
    }

    part systemInstance : System {
        part c1 :> confidentialComponent;
        part c2 :> unclassifiedComponent;
        part c3 :> secretComponent;
    }

}