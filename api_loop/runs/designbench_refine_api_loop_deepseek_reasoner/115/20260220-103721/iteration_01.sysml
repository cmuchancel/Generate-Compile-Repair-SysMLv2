package AutomotiveConfiguration {
    private import ScalarValues::*;

    part def Component1;
    part def Component2;
    part def Component3;
    part def Component4;
    part def Component5;
    part def Component6;

    part def SubsystemA_Config1 {
        part comp1 : Component1;
        part comp2 : Component2;
    }

    part def SubsystemA_Config2 {
        part comp2 : Component2;
        part comp3 : Component3;
    }

    part def SubsystemB_Config1 {
        part comp4 : Component4;
        part comp5 : Component5;
    }

    part def SubsystemB_Config2 {
        part comp5 : Component5;
        part comp6 : Component6;
    }

    part def Automobile {
        part subsystemA : SubsystemA_Config1; 
        part subsystemB : SubsystemB_Config1; 
    }

    part def ConfigurationA {
        part automobile : Automobile {
            part subsystemA :> SubsystemA_Config1;
            part subsystemB :> SubsystemB_Config1;
        }
    }

    part def ConfigurationB {
        part automobile : Automobile {
            part subsystemA :> SubsystemA_Config2;
            part subsystemB :> SubsystemB_Config1;
        }
    }

    part ConfigurationA_instance : ConfigurationA;
    part ConfigurationB_instance : ConfigurationB;
}