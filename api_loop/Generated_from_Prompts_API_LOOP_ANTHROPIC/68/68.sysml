package VehiclePowerSystem {
    private import ScalarValues::*;

    enum def GearState {
        enum NEUTRAL;
        enum FIRST;
        enum SECOND;
        enum THIRD;
        enum FOURTH;
        enum FIFTH;
    }

    enum def ClutchState {
        enum ENGAGED;
        enum DISENGAGED;
        enum SLIPPING;
    }

    item def FuelItem {
        attribute fuelMass : Real;
    }

    item def TorqueSignal {
        attribute torqueValue : Real;
        attribute speedValue : Real;
    }

    port def FuelPort {
        attribute massFlow : Real;
    }

    port def TorquePort {
        attribute torque : Real;
        attribute angularSpeed : Real;
    }

    port def PowerPort {
        attribute power : Real;
        attribute speed : Real;
    }

    part def FuelTank {
        attribute fuelMass : Real = 50.0;
        attribute maxFuelMass : Real = 80.0;
        attribute minFuelMass : Real = 0.1;
        port fuelOut : FuelPort;
    }

    part def Engine {
        attribute idleSpeed : Real = 800.0;
        attribute maxSpeed : Real = 6500.0;
        attribute peakTorque : Real = 350.0;
        attribute peakTorqueSpeed : Real = 3500.0;
        attribute currentSpeed : Real = 1000.0;
        attribute currentTorque : Real = 200.0;
        attribute torqueAtLowSpeed : Real = 150.0;
        attribute torqueAtMidSpeed : Real = 350.0;
        attribute torqueAtHighSpeed : Real = 180.0;
        port fuelIn : FuelPort;
        port torqueOut : TorquePort;
    }

    part def Clutch {
        attribute clutchState : ClutchState = ClutchState::ENGAGED;
        attribute maxTransmittableTorque : Real = 400.0;
        attribute inputTorque : Real = 200.0;
        attribute outputTorque : Real = 200.0;
        attribute engagementRatio : Real = 1.0;
        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
    }

    part def Transmission {
        attribute currentGear : GearState = GearState::FIRST;
        attribute gearRatio1 : Real = 3.5;
        attribute gearRatio2 : Real = 2.1;
        attribute gearRatio3 : Real = 1.4;
        attribute gearRatio4 : Real = 1.0;
        attribute gearRatio5 : Real = 0.8;
        attribute inputTorque : Real = 200.0;
        attribute outputTorque : Real = 700.0;
        attribute efficiency : Real = 0.95;
        port torqueIn : TorquePort;
        port powerOut : PowerPort;
    }

    part def DriveShaft {
        attribute inputPower : Real;
        attribute outputPower : Real;
        attribute efficiency : Real = 0.98;
        port powerIn : PowerPort;
        port powerOut : PowerPort;
    }

    part def Wheel {
        attribute radius : Real = 0.32;
        attribute inputPower : Real;
        attribute tractionForce : Real;
        port powerIn : PowerPort;
    }

    part def Vehicle {
        attribute curbWeight : Real = 1500.0;
        attribute fuelMass : Real = 50.0;
        attribute maxMassLimit : Real = 1650.0;
        attribute totalMass : Real = 1550.0;
        attribute massConstraintSatisfied : Boolean = true;
        attribute fuelMassPositive : Boolean = true;

        part tank : FuelTank;
        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;
        part driveShaft : DriveShaft;
        part frontLeftWheel : Wheel;
        part frontRightWheel : Wheel;

        connect tank.fuelOut to engine.fuelIn;
        connect engine.torqueOut to clutch.torqueIn;
        connect clutch.torqueOut to transmission.torqueIn;
        connect transmission.powerOut to driveShaft.powerIn;
        connect driveShaft.powerOut to frontLeftWheel.powerIn;
        connect driveShaft.powerOut to frontRightWheel.powerIn;
    }

    part vehicleSystem : Vehicle;
}