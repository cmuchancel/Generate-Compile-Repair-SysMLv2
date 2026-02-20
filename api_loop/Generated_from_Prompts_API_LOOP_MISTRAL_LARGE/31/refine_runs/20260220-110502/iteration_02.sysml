package VehicleSimulation {
    private import ScalarValues::*;

    enum def TimeUnit {
        enum second;
    }

    enum def DistanceUnit {
        enum meter;
    }

    enum def MassUnit {
        enum kilogram;
    }

    enum def PowerUnit {
        enum watt;
    }

    enum def VelocityUnit {
        enum meterPerSecond;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute massUnit : MassUnit = MassUnit::kilogram;
        attribute initialPosition : Real;
        attribute positionUnit : DistanceUnit = DistanceUnit::meter;
        attribute initialVelocity : Real;
        attribute velocityUnit : VelocityUnit = VelocityUnit::meterPerSecond;
    }

    part def SimulationParameters {
        attribute timeStep : Real;
        attribute timeUnit : TimeUnit = TimeUnit::second;
    }

    part def PowerInput {
        attribute powerSequence : Sequence<Real>;
        attribute powerUnit : PowerUnit = PowerUnit::watt;
    }

    part def SimulationOutput {
        attribute positionTrajectory : Sequence<Real>;
        attribute positionUnit : DistanceUnit = DistanceUnit::meter;
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