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
        attribute component1 : Component;
        attribute component2 : Component;
        attribute component3 : Component;
    }
    
    part exampleSystem : System {
        attribute component1 : Component = Component {
            securityClassification = SecurityLevel::Confidential
        };
        attribute component2 : Component = Component {
            securityClassification = SecurityLevel::Unclassified
        };
        attribute component3 : Component = Component {
            securityClassification = SecurityLevel::Secret
        };
    }
}