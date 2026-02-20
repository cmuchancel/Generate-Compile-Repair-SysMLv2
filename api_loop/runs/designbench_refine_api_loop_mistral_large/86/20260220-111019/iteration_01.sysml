package VehicleDynamicsSimulation {
    private import ScalarValues::*;

    enum def TimeUnit {
        enum second;
        enum millisecond;
    }

    enum def DistanceUnit {
        enum meter;
        enum kilometer;
    }

    enum def VelocityUnit {
        enum meterPerSecond;
        enum kilometerPerHour;
    }

    part def Vehicle {
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
        attribute timeStepUnit : TimeUnit = TimeUnit::second;
    }

    part def SimulationController {
        attribute simulationTime : Real;
        attribute simulationTimeUnit : TimeUnit = TimeUnit::second;
    }

    part def ModelCenterInterface {
        attribute isConnected : Boolean = false;
    }

    part def SimulationOutput {
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
        attribute velocityUnit : VelocityUnit = VelocityUnit::meterPerSecond;
        attribute distanceUnit : DistanceUnit = DistanceUnit::meter;
    }

    part system : VehicleDynamicsSimulation {
        part vehicle : Vehicle;
        part controller : SimulationController;
        part modelCenter : ModelCenterInterface;
        part output : SimulationOutput;

        connect vehicle to controller;
        connect controller to modelCenter;
        connect modelCenter to output;
    }
}