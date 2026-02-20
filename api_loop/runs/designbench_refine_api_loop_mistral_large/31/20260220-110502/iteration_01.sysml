package VehicleSimulation {
    private import ScalarValues::*;

    enum def TimeUnit {
        second
    }

    enum def DistanceUnit {
        meter
    }

    enum def MassUnit {
        kilogram
    }

    enum def PowerUnit {
        watt
    }

    enum def VelocityUnit {
        meterPerSecond
    }

    part def Vehicle {
        attribute mass : Real;
        attribute massUnit : MassUnit = kilogram;
        attribute initialPosition : Real;
        attribute positionUnit : DistanceUnit = meter;
        attribute initialVelocity : Real;
        attribute velocityUnit : VelocityUnit = meterPerSecond;
    }

    part def SimulationParameters {
        attribute timeStep : Real;
        attribute timeUnit : TimeUnit = second;
    }

    part def PowerInput {
        attribute powerSequence : Sequence<Real>;
        attribute powerUnit : PowerUnit = watt;
    }

    part def SimulationOutput {
        attribute positionTrajectory : Sequence<Real>;
        attribute positionUnit : DistanceUnit = meter;
    }

    part def Simulation {
        part vehicle : Vehicle;
        part parameters : SimulationParameters;
        part powerInput : PowerInput;
        part output : SimulationOutput;

        port def PowerPort {
            attribute power : Real;
        }

        port powerIn : PowerPort;
        port powerOut : PowerPort;

        connect powerInput to powerIn;
        connect powerOut to output;
    }
}