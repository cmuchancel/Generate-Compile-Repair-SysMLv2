package VehicleDynamics {
    private import ScalarValues::*;
    
    part def VehicleDynamicsSystem {
        port in inputPort : InputData;
        port out outputPort : OutputData;
        
        part controller : VehicleController;
        part dynamics : VehicleDynamicsModel;
        
        connect controller.commandPort to dynamics.controlPort;
        connect inputPort to controller.inputPort;
        connect dynamics.outputPort to outputPort;
    }
    
    part def VehicleController {
        port in inputPort : InputData;
        port out commandPort : ControlCommand;
        
        attribute operatingMode : OperatingMode = OperatingMode::Normal;
        attribute controlAlgorithm : ControlAlgorithmType = ControlAlgorithmType::Basic;
    }
    
    part def VehicleDynamicsModel {
        port in controlPort : ControlCommand;
        port out outputPort : OutputData;
        
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
        port in controlInput : ControlCommand;
        port out powerOutput : PowerOutput;
        
        attribute wheelPower : Real;
        attribute aerodynamicDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
        attribute availablePower : Real;
    }
    
    part def MotionCalculationModule {
        port in controlInput : ControlCommand;
        port in powerInput : PowerOutput;
        port out accelerationOutput : AccelerationOutput;
        port out velocityOutput : VelocityOutput;
        
        attribute effectivePower : Real;
        attribute acceleration : Real;
        attribute currentVelocity : Real;
    }
    
    part def IntegrationModule {
        port in accelerationInput : AccelerationOutput;
        port in velocityInput : VelocityOutput;
        port out positionOutput : PositionOutput;
        port out velocityOutput : VelocityOutput;
        
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