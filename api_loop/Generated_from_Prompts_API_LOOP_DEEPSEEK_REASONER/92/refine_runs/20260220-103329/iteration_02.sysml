package StructuralDecompositionExample {
    private import ScalarValues::*;

    port def PowerPort {
        attribute voltage : Real;
        attribute maxCurrent : Real;
    }

    port def DataPort {
        attribute protocol : String;
        attribute bandwidth : Real;
    }

    port def MechanicalPort {
        attribute maxLoad : Real;
    }

    part def Battery {
        port p : PowerPort;
        attribute capacity : Real;
    }

    part def SolarPanel {
        port p : PowerPort;
        attribute efficiency : Real;
    }

    part def PowerBus {
        port input : PowerPort;
        port output : PowerPort;
    }

    part def PowerSubsystem {
        part battery : Battery;
        part solarPanel : SolarPanel;
        part bus : PowerBus;
        connect battery.p to bus.input;
        connect solarPanel.p to bus.input;
    }

    part def Sensor {
        port data : DataPort;
        port power : PowerPort;
        attribute accuracy : Real;
    }

    part def Processor {
        port dataIn : DataPort;
        port dataOut : DataPort;
        port power : PowerPort;
        attribute speed : Real;
    }

    part def CommunicationSubsystem {
        part sensor : Sensor;
        part processor : Processor;
        connect sensor.data to processor.dataIn;
        connect sensor.power to processor.power;
    }

    part def StructuralFrame {
        port mount : MechanicalPort;
        attribute weight : Real;
    }

    part def ThermalSystem {
        port power : PowerPort;
        attribute coolingCapacity : Real;
    }

    part def MainSystem {
        part powerSubsystem : PowerSubsystem;
        part commSubsystem : CommunicationSubsystem;
        part structuralFrame : StructuralFrame;
        part thermal : ThermalSystem;

        connect powerSubsystem.bus.output to commSubsystem.processor.power;
        connect powerSubsystem.bus.output to thermal.power;
    }
}