package VehicleDynamics {
    private import ScalarValues::*;

    part def VehicleDynamicsSystem {
        port inputPort : InputData;
        port outputPort : OutputData;

        part controller : VehicleController;
        part dynamics : VehicleDynamicsModel;

        connect inputPort to controller.inputPort;
        connect controller.commandPort to dynamics.controlPort;
        connect dynamics.outputPort to outputPort;
    }

    part def VehicleController {
        port inputPort : InputData;
        port commandPort : ControlCommand;

        attribute operatingMode : OperatingMode = OperatingMode::Normal;
        attribute controlAlgorithm : ControlAlgorithmType = ControlAlgorithmType::Basic;
    }

    part def VehicleDynamicsModel {
        port controlPort : ControlCommand;
        port outputPort : OutputData;

        part powerModule : PowerCalculationModule;
        part motionModule : MotionCalculationModule;
        part integrationModule : IntegrationModule;

        connect controlPort to powerModule.controlInput;
        connect controlPort to motionModule.controlInput;
        connect powerModule.powerOutput to motionModule.powerInput;
        connect motionModule.accelerationOutput to integrationModule.accelerationInput;
        connect motionModule.velocityOutput to integrationModule.velocityInput;
        connect integrationModule.positionOutput to outputPort;
        connect integrationModule.velocityOutput to outputPort;
        connect motionModule.accelerationOutput to outputPort;
        connect powerModule.powerOutput to outputPort;
    }

    part def PowerCalculationModule {
        port controlInput : ControlCommand;
        port powerOutput : PowerOutput;

        attribute wheelPower : Real;
        attribute aerodynamicDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
        attribute availablePower : Real;
    }

    part def MotionCalculationModule {
        port controlInput : ControlCommand;
        port powerInput : PowerOutput;
        port accelerationOutput : AccelerationOutput;
        port velocityOutput : VelocityOutput;

        attribute effectivePower : Real;
        attribute acceleration : Real;
        attribute currentVelocity : Real;
    }

    part def IntegrationModule {
        port accelerationInput : AccelerationOutput;
        port velocityInput : VelocityOutput;
        port positionOutput : PositionOutput;
        port velocityOutput : VelocityOutput;

        attribute initialVelocity : Real = 0.0;
        attribute initialPosition : Real = 0.0;
        attribute timeStep : Real = 0.1;
        attribute currentPosition : Real;
        attribute currentVelocity : Real;
    }

    item def InputData {
        attribute wheelPower : Real;
        attribute aerodynamicDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute vehicleMass : Real;
        attribute initialSpeed : Real;
        attribute simulationTime : Real;
    }

    item def OutputData {
        attribute availablePower : Real;
        attribute acceleration : Real;
        attribute velocity : Real;
        attribute position : Real;
        attribute calculationTime : Real;
    }

    item def ControlCommand {
        attribute calculationMode : CalculationModeType = CalculationModeType::Full;
        attribute timeStepSize : Real = 0.1;
    }

    item def PowerOutput {
        attribute value : Real;
        attribute calculationTime : Real;
    }

    item def AccelerationOutput {
        attribute value : Real;
        attribute calculationTime : Real;
    }

    item def VelocityOutput {
        attribute value : Real;
        attribute calculationTime : Real;
    }

    item def PositionOutput {
        attribute value : Real;
        attribute calculationTime : Real;
    }

    enum def OperatingMode {
        enum Normal;
        enum Performance;
        enum Economy;
    }

    enum def ControlAlgorithmType {
        enum Basic;
        enum Advanced;
        enum Adaptive;
    }

    enum def CalculationModeType {
        enum PowerOnly;
        enum MotionOnly;
        enum Full;
    }
}