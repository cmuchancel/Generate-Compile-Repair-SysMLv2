package VehicleLongitudinalDynamics {

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
        enum meter_per_second;
        enum kilometer_per_hour;
    }

    enum def AccelerationUnit {
        enum meter_per_second_squared;
    }

    enum def MassUnit {
        enum kilogram;
        enum tonne;
    }

    enum def PowerUnit {
        enum watt;
        enum kilowatt;
    }

    part def VehicleParameters {
        attribute mass : Real;
        attribute massUnit : MassUnit = MassUnit::kilogram;
        attribute initialPosition : Real;
        attribute positionUnit : DistanceUnit = DistanceUnit::meter;
        attribute initialVelocity : Real;
        attribute velocityUnit : VelocityUnit = VelocityUnit::meter_per_second;
    }

    part def PowerSegment {
        attribute startTime : Real;
        attribute endTime : Real;
        attribute powerValue : Real;
        attribute powerUnit : PowerUnit = PowerUnit::watt;
        attribute timeUnit : TimeUnit = TimeUnit::second;
    }

    part def SimulationSettings {
        attribute timeStep : Real;
        attribute timeUnit : TimeUnit = TimeUnit::second;
    }

    part def SimulationOutput {
        attribute accelerations : Real[0..*];
        attribute velocities : Real[0..*];
        attribute positions : Real[0..*];
        attribute accelerationUnit : AccelerationUnit = AccelerationUnit::meter_per_second_squared;
        attribute velocityUnit : VelocityUnit = VelocityUnit::meter_per_second;
        attribute positionUnit : DistanceUnit = DistanceUnit::meter;
    }

    part def LongitudinalDynamicsSimulator {
        part parameters : VehicleParameters;
        part powerSegments : PowerSegment[0..*];
        part settings : SimulationSettings;
        part output : SimulationOutput;
    }

    item def PowerData {
        attribute segments : PowerSegment[0..*];
    }

    item def MotionData {
        attribute accelerations : Real[0..*];
        attribute velocities : Real[0..*];
        attribute positions : Real[0..*];
    }

    port def PowerInputPort {
        in power : PowerData;
    }

    port def MotionOutputPort {
        out motion : MotionData;
    }

    part simulator : LongitudinalDynamicsSimulator;

    connect simulator.parameters to simulator;
    connect simulator.powerSegments to simulator;
    connect simulator.settings to simulator;
    connect simulator.output to simulator;
}