package AutomotiveConfigurationSystem {
    private import ScalarValues::*;

    enum def SubsystemAChoice { Config1; Config2; }
    enum def SubsystemBChoice { Config3; Config4; }

    part def Component1;
    part def Component2;
    part def Component3;
    part def Component4;
    part def Component5;
    part def Component6;

    part def SubsystemA;
    part def SubsystemA_Config1 {
        part comp1 : Component1;
        part comp2 : Component2;
        attribute choice : SubsystemAChoice = SubsystemAChoice::Config1;
    }
    part def SubsystemA_Config2 {
        part comp2 : Component2;
        part comp3 : Component3;
        attribute choice : SubsystemAChoice = SubsystemAChoice::Config2;
    }

    part def SubsystemB;
    part def SubsystemB_Config3 {
        part comp4 : Component4;
        part comp5 : Component5;
        attribute choice : SubsystemBChoice = SubsystemBChoice::Config3;
    }
    part def SubsystemB_Config4 {
        part comp5 : Component5;
        part comp6 : Component6;
        attribute choice : SubsystemBChoice = SubsystemBChoice::Config4;
    }

    part def Automobile {
        part subA : SubsystemA;
        part subB : SubsystemB;
        attribute subAChoice : SubsystemAChoice;
        attribute subBChoice : SubsystemBChoice;
        attribute constraintOk : Boolean = (subAChoice != SubsystemAChoice::Config2) or (subBChoice == SubsystemBChoice::Config3);
    }

    part def ConfigurationA {
        part subA : SubsystemA_Config1;
        part subB : SubsystemB_Config3;
        attribute subAChoice : SubsystemAChoice = SubsystemAChoice::Config1;
        attribute subBChoice : SubsystemBChoice = SubsystemBChoice::Config3;
        attribute constraintOk : Boolean = true;
    }

    part def ConfigurationB {
        part subA : SubsystemA_Config2;
        part subB : SubsystemB_Config3;
        attribute subAChoice : SubsystemAChoice = SubsystemAChoice::Config2;
        attribute subBChoice : SubsystemBChoice = SubsystemBChoice::Config3;
        attribute constraintOk : Boolean = true;
    }
}