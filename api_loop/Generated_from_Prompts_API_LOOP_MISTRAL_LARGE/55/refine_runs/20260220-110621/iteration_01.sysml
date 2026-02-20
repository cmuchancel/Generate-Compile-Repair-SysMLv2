package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassFilterType {
        enum ALL
        enum THRESHOLD
    }

    part def Vehicle {
        attribute vin : String;
        attribute baseMass : Real;
        attribute totalMass : Real;
        part components : Component[*];
    }

    part def Component {
        attribute serialNumber : String;
        attribute mass : Real;
        part subcomponents : Component[*];
    }

    part def Engine : Component {
    }

    part def Transmission : Component {
    }

    part def MassCalculator {
        attribute filterType : MassFilterType = MassFilterType::ALL;
        attribute massThreshold : Real = 0.0;

        port calculateMass : MassCalculationPort;
    }

    port def MassCalculationPort {
    }

    item def MassCalculationRequest {
        attribute vehicle : Vehicle;
        attribute filter : MassFilterType;
        attribute threshold : Real;
    }

    item def MassCalculationResult {
        attribute totalMass : Real;
    }

    part vehicle : Vehicle;
    part calculator : MassCalculator;

    connect calculator.calculateMass to vehicle;
}