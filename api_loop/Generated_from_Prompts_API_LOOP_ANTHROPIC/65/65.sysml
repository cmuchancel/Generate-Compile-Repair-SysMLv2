package VehicleWeightManagement {
    private import ScalarValues::*;

    enum def WeightStatus {
        enum OK;
        enum WARNING;
    }

    item def WeightData {
        attribute value : Real;
    }

    port def WeightInputPort {
        attribute weightValue : Real;
    }

    part def Chassis {
        attribute weight : Real = 800.0;
        port weightOut : WeightInputPort;
    }

    part def Engine {
        attribute weight : Real = 600.0;
        port weightOut : WeightInputPort;
    }

    part def Transmission {
        attribute weight : Real = 400.0;
        port weightOut : WeightInputPort;
    }

    part def WeightAggregator {
        attribute chassisWeight : Real;
        attribute engineWeight : Real;
        attribute transmissionWeight : Real;
        attribute totalWeight : Real;
        attribute weightLimit : Real = 2500.0;
        attribute status : WeightStatus;
        port chassisIn : WeightInputPort;
        port engineIn : WeightInputPort;
        port transmissionIn : WeightInputPort;
    }

    part def VehicleWeightSystem {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        part aggregator : WeightAggregator;

        connect chassis.weightOut to aggregator.chassisIn;
        connect engine.weightOut to aggregator.engineIn;
        connect transmission.weightOut to aggregator.transmissionIn;
    }

    part vehicleSystem : VehicleWeightSystem;
}