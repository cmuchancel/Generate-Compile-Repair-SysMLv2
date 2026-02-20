package AutomotiveConfigurationManagement {
    private import ScalarValues::*;

    enum def SubsystemAConfig {
        enum Config1;
        enum Config2;
    }

    enum def SubsystemBConfig {
        enum Config1;
        enum Config2;
    }

    part def Component1;
    part def Component2;
    part def Component3;
    part def Component4;
    part def Component5;
    part def Component6;

    part def SubsystemA {
        attribute config : SubsystemAConfig;

        part comp1 : Component1;
        part comp2 : Component2;
        part comp3 : Component3;
    }

    part def SubsystemB {
        attribute config : SubsystemBConfig;

        part comp4 : Component4;
        part comp5 : Component5;
        part comp6 : Component6;
    }

    part def Automobile {
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;

        attribute vehicleConfig : String;
    }

    part automobileA : Automobile {
        part subsystemA : SubsystemA {
            attribute config : SubsystemAConfig = SubsystemAConfig::Config1;
        }
        part subsystemB : SubsystemB {
            attribute config : SubsystemBConfig = SubsystemBConfig::Config1;
        }
        attribute vehicleConfig : String = "Configuration A";
    }

    part automobileB : Automobile {
        part subsystemA : SubsystemA {
            attribute config : SubsystemAConfig = SubsystemAConfig::Config2;
        }
        part subsystemB : SubsystemB {
            attribute config : SubsystemBConfig = SubsystemBConfig::Config1;
        }
        attribute vehicleConfig : String = "Configuration B";
    }
}