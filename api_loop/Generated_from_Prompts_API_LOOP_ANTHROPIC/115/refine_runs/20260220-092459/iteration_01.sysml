package AutomobileConfigurationManagement {
    private import ScalarValues::*;

    enum def SubsystemAConfig {
        enum CONFIG_A1;
        enum CONFIG_A2;
    }

    enum def SubsystemBConfig {
        enum CONFIG_B1;
        enum CONFIG_B2;
    }

    part def Component1;
    part def Component2;
    part def Component3;
    part def Component4;
    part def Component5;
    part def Component6;

    part def SubsystemAOption1 {
        part comp1 : Component1;
        part comp2 : Component2;
    }

    part def SubsystemAOption2 {
        part comp2 : Component2;
        part comp3 : Component3;
    }

    part def SubsystemBOption1 {
        part comp4 : Component4;
        part comp5 : Component5;
    }

    part def SubsystemBOption2 {
        part comp5 : Component5;
        part comp6 : Component6;
    }

    part def SubsystemA {
        attribute selectedConfig : SubsystemAConfig;
        part optionA1 : SubsystemAOption1;
        part optionA2 : SubsystemAOption2;
    }

    part def SubsystemB {
        attribute selectedConfig : SubsystemBConfig;
        part optionB1 : SubsystemBOption1;
        part optionB2 : SubsystemBOption2;
    }

    part def Automobile {
        attribute subsystemASelection : SubsystemAConfig;
        attribute subsystemBSelection : SubsystemBConfig;
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;
    }

    part def ConfigurationA {
        attribute subsystemASelection : SubsystemAConfig = SubsystemAConfig::CONFIG_A1;
        attribute subsystemBSelection : SubsystemBConfig = SubsystemBConfig::CONFIG_B1;
        part subsystemA : SubsystemAOption1;
        part subsystemB : SubsystemBOption1;
    }

    part def ConfigurationB {
        attribute subsystemASelection : SubsystemAConfig = SubsystemAConfig::CONFIG_A2;
        attribute subsystemBSelection : SubsystemBConfig = SubsystemBConfig::CONFIG_B1;
        part subsystemA : SubsystemAOption2;
        part subsystemB : SubsystemBOption1;
    }

    part vehicleConfigA : ConfigurationA;
    part vehicleConfigB : ConfigurationB;

    part automobile : Automobile {
        attribute subsystemASelection : SubsystemAConfig = SubsystemAConfig::CONFIG_A1;
        attribute subsystemBSelection : SubsystemBConfig = SubsystemBConfig::CONFIG_B1;
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;
    }
}