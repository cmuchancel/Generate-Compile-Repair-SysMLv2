package VehiclePowertrainOptimization {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
        enum EightCylinder;
        enum Hybrid;
        enum Electric;
    }

    part def Cylinder;
    part def Piston;
    part def ConnectingRod;
    part def Crankshaft;
    part def EngineBlock;
    part def EngineControlUnit;

    part def Engine {
        attribute engineType : EngineType;
        attribute power : Real;
        attribute mass : Real;
        attribute efficiency : Real;
        attribute reliability : Real;
        attribute cost : Real;

        part cylinders : Cylinder[4..8];
        part pistons : Piston[4..8];
        part connectingRods : ConnectingRod[4..8];
        part crankshaft : Crankshaft;
        part engineBlock : EngineBlock;
        part ecu : EngineControlUnit;
    }

    part def Vehicle {
        part engine : Engine;
    }

    part def TradeOffAnalysis {
        attribute powerWeight : Real = 0.2;
        attribute massWeight : Real = 0.2;
        attribute efficiencyWeight : Real = 0.2;
        attribute reliabilityWeight : Real = 0.2;
        attribute costWeight : Real = 0.2;

        attribute evaluate : Real;
    }

    part def PowertrainOptimizer {
        part vehicle : Vehicle;
        part tradeOffAnalysis : TradeOffAnalysis;

        attribute recommendOptimalEngine : Engine;
    }

    port def InputPort;
    port def OutputPort;

    part def UserInterface {
        port input : InputPort;
        port output : OutputPort;
    }

    part system : PowertrainOptimizer {
        part ui : UserInterface;
    }

    connect system.ui.output to system.tradeOffAnalysis;
}