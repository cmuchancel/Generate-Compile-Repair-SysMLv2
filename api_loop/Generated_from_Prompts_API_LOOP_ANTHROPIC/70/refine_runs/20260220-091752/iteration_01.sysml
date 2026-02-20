package VehicleSystem {
    private import ScalarValues::*;

    enum def FuelState {
        enum FullyFueled;
        enum Empty;
    }

    item def Torque;
    item def MechanicalPower;

    port def ClutchPort {
        attribute torqueValue : Real;
        attribute speedValue : Real;
    }

    part def Engine {
        attribute idleSpeedRPM : Real = 800.0;
        attribute maxSpeedRPM : Real = 6000.0;
        attribute torqueAtIdle : Real = 150.0;
        attribute torqueAtPeak : Real = 400.0;
        attribute torqueAtMax : Real = 250.0;
        attribute peakTorqueRPM : Real = 3500.0;

        port clutchOutput : ClutchPort;
    }

    part def Transmission {
        port clutchInput : ClutchPort;
    }

    part def Clutch {
        port engineSide : ClutchPort;
        port transmissionSide : ClutchPort;
    }

    part def Vehicle {
        attribute maxFueledMassKg : Real = 2000.0;
        attribute maxEmptyMassKg : Real = 1500.0;
        attribute fueledMassKg : Real = 1800.0;
        attribute emptyMassKg : Real = 1300.0;
        attribute fueledMassWithinLimit : Boolean = true;
        attribute emptyMassWithinLimit : Boolean = true;

        part engine : Engine;
        part transmission : Transmission;
        part clutch : Clutch;

        connect engine.clutchOutput to clutch.engineSide;
        connect clutch.transmissionSide to transmission.clutchInput;
    }
}