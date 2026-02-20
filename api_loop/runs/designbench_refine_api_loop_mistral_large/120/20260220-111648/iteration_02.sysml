package VehicleQualityAnalysis {
    private import ScalarValues::*;

    enum def QualityAttribute {
        mass,
        durability,
        reliability,
        performance,
        efficiency;
    }

    part def Vehicle {
        attribute totalMass : Real;
        part engine : Engine;
        part transmission : Transmission;
        part frontAxle : AxleAssembly;
        part rearAxle : AxleAssembly;
    }

    part def Engine {
        attribute mass : Real;
        attribute durability : Real;
        attribute reliability : Real;
    }

    part def Transmission {
        attribute mass : Real;
        attribute efficiency : Real;
    }

    part def AxleAssembly {
        attribute mass : Real;
        attribute durability : Real;
    }

    part def QualityAnalysisSystem {
        part vehicle : Vehicle;
        attribute analysisObjective : String;
        attribute analysisPlan : String;
        attribute totalMass : Real;
    }

    part def QualityDataInput {
        port inputData : QualityDataPort;
    }

    port def QualityDataPort {
    }

    item def QualityData {
        attribute componentName : String;
        attribute attributeType : QualityAttribute;
        attribute value : Real;
    }

    part qualitySystem : QualityAnalysisSystem;
    part dataInput : QualityDataInput;

    connect dataInput.inputData to qualitySystem.vehicle;
}