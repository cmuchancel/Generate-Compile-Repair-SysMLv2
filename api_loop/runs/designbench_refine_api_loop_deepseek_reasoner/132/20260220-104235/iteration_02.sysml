package SecurityManagement {

    private import ScalarValues::*;

    enum def SecurityLevel {
        enum Unclassified;
        enum Confidential;
        enum Secret;
    }

    part def Component {
        attribute securityClassification : SecurityLevel;
    }

    part def System {
        part component1 : Component;
        part component2 : Component;
        part component3 : Component;
    }

    part exampleSystem : System {
        part component1 : Component {
            attribute securityClassification : SecurityLevel = SecurityLevel::Confidential;
        };
        part component2 : Component {
            attribute securityClassification : SecurityLevel = SecurityLevel::Unclassified;
        };
        part component3 : Component {
            attribute securityClassification : SecurityLevel = SecurityLevel::Secret;
        };
    }
}