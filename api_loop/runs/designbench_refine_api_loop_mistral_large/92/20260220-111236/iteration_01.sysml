package StructuralDecomposition {
    private import ScalarValues::*;

    part def System {
        attribute name : String = "TopLevelSystem";
        part subsystem1 : Subsystem;
        part subsystem2 : Subsystem;
        part subsystem3 : Subsystem;

        port def SystemPort {
            attribute protocol : String;
        }

        port systemPort1 : SystemPort;
        port systemPort2 : SystemPort;
    }

    part def Subsystem {
        attribute name : String;
        part component1 : Component;
        part component2 : Component;

        port def SubsystemPort {
            attribute protocol : String;
        }

        port subsystemPort1 : SubsystemPort;
        port subsystemPort2 : SubsystemPort;
    }

    part def Component {
        attribute name : String;
        attribute function : String;

        port def ComponentPort {
            attribute protocol : String;
        }

        port componentPort1 : ComponentPort;
        port componentPort2 : ComponentPort;
    }

    part topSystem : System;

    connect topSystem.subsystem1.subsystemPort1 to topSystem.subsystem2.subsystemPort1;
    connect topSystem.subsystem2.subsystemPort2 to topSystem.subsystem3.subsystemPort1;
    connect topSystem.subsystem1.component1.componentPort1 to topSystem.subsystem1.component2.componentPort1;
    connect topSystem.subsystem2.component1.componentPort1 to topSystem.subsystem2.component2.componentPort1;
    connect topSystem.subsystem3.component1.componentPort1 to topSystem.subsystem3.component2.componentPort1;
}