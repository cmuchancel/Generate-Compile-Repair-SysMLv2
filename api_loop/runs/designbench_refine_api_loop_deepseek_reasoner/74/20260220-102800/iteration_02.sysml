package EngineSelectionSystem {
    private import ScalarValues::*;

    enum def EngineType {
        enum FOUR_CYLINDER,
        enum SIX_CYLINDER
    }

    part def Engine {
        attribute type : EngineType;
        attribute power : Real;
        attribute mass : Real;
        attribute efficiency : Real;
        attribute cost : Real;
        attribute score : Real;
    }

    part def SelectionSystem {
        part engines : Engine[0..*];
        part selectedEngine : Engine;
        attribute bestScore : Real = 0.0;
    }

    part engine1 : Engine;
    part engine2 : Engine;
    part engine3 : Engine;
    part system : SelectionSystem;

    connect engine1 to system.engines;
    connect engine2 to system.engines;
    connect engine3 to system.engines;
}