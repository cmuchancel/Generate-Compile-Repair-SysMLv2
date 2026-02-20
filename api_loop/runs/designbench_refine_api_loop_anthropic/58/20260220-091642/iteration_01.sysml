package VehicleDynamics {
    private import ScalarValues::*;

    part def VehicleDynamicsSystem {
        attribute wheelPower : Real;
        attribute aeroDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real;
        attribute gravitationalAcceleration : Real;

        attribute aeroDragForce : Real;
        attribute rollingResistanceForce : Real;
        attribute totalResistanceForce : Real;
        attribute resistancePower : Real;
        attribute availablePower : Real;

        attribute effectivePower : Real;
        attribute acceleration : Real;

        attribute initialSpeed : Real;
        attribute elapsedTime : Real;
        attribute speedAtTime : Real;

        attribute displacement : Real;
    }

    part def PowerCalculator {
        attribute wheelPower : Real;
        attribute aeroDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real;
        attribute gravitationalAcceleration : Real;

        attribute aeroDragForce : Real;
        attribute rollingResistanceForce : Real;
        attribute totalResistanceForce : Real;
        attribute resistancePower : Real;
        attribute availablePower : Real;
    }

    part def AccelerationCalculator {
        attribute effectivePower : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
        attribute netForce : Real;
        attribute acceleration : Real;
    }

    part def SpeedCalculator {
        attribute initialSpeed : Real;
        attribute acceleration : Real;
        attribute elapsedTime : Real;
        attribute speedAtTime : Real;
    }

    part def DisplacementCalculator {
        attribute initialSpeed : Real;
        attribute acceleration : Real;
        attribute elapsedTime : Real;
        attribute displacement : Real;
    }

    port def PowerInputPort {
        attribute wheelPower : Real;
        attribute aeroDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real;
        attribute gravitationalAcceleration : Real;
    }

    port def PowerOutputPort {
        attribute availablePower : Real;
        attribute effectivePower : Real;
    }

    port def AccelerationInputPort {
        attribute effectivePower : Real;
        attribute vehicleMass : Real;
        attribute currentSpeed : Real;
    }

    port def AccelerationOutputPort {
        attribute acceleration : Real;
    }

    port def SpeedInputPort {
        attribute initialSpeed : Real;
        attribute acceleration : Real;
        attribute elapsedTime : Real;
    }

    port def SpeedOutputPort {
        attribute speedAtTime : Real;
    }

    port def DisplacementInputPort {
        attribute initialSpeed : Real;
        attribute acceleration : Real;
        attribute elapsedTime : Real;
    }

    port def DisplacementOutputPort {
        attribute displacement : Real;
    }

    part def VehicleDynamicsAssembly {
        part powerCalc : PowerCalculator {
            port powerIn : PowerInputPort;
            port powerOut : PowerOutputPort;
            attribute wheelPower : Real = 150000.0;
            attribute aeroDragCoefficient : Real = 0.3;
            attribute frictionCoefficient : Real = 0.015;
            attribute vehicleMass : Real = 1500.0;
            attribute currentSpeed : Real = 20.0;
            attribute frontalArea : Real = 2.2;
            attribute airDensity : Real = 1.225;
            attribute gravitationalAcceleration : Real = 9.81;
            attribute aeroDragForce : Real = 0.0;
            attribute rollingResistanceForce : Real = 0.0;
            attribute totalResistanceForce : Real = 0.0;
            attribute resistancePower : Real = 0.0;
            attribute availablePower : Real = 0.0;
        }

        part accelCalc : AccelerationCalculator {
            port accelIn : AccelerationInputPort;
            port accelOut : AccelerationOutputPort;
            attribute effectivePower : Real = 0.0;
            attribute vehicleMass : Real = 1500.0;
            attribute currentSpeed : Real = 20.0;
            attribute netForce : Real = 0.0;
            attribute acceleration : Real = 0.0;
        }

        part speedCalc : SpeedCalculator {
            port speedIn : SpeedInputPort;
            port speedOut : SpeedOutputPort;
            attribute initialSpeed : Real = 20.0;
            attribute acceleration : Real = 0.0;
            attribute elapsedTime : Real = 5.0;
            attribute speedAtTime : Real = 0.0;
        }

        part dispCalc : DisplacementCalculator {
            port dispIn : DisplacementInputPort;
            port dispOut : DisplacementOutputPort;
            attribute initialSpeed : Real = 20.0;
            attribute acceleration : Real = 0.0;
            attribute elapsedTime : Real = 5.0;
            attribute displacement : Real = 0.0;
        }

        connect powerCalc.powerOut to accelCalc.accelIn;
        connect accelCalc.accelOut to speedCalc.speedIn;
        connect accelCalc.accelOut to dispCalc.dispIn;
    }

    part vehicleSystem : VehicleDynamicsAssembly;
}