package VehicleSystem {
    private import ScalarValues::*;
    
    part def Chassis {
        attribute mass : Real;
    }
    
    part def Engine {
        attribute mass : Real;
    }
    
    part def Transmission {
        attribute mass : Real;
    }
    
    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        attribute totalMass : Real;
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute appliedForce : Real;
        attribute time : Real;
    }
    
    item def MassConstraint {
        attribute totalMass : Real;
        attribute chassisMass : Real;
        attribute engineMass : Real;
        attribute transmissionMass : Real;
    }
    
    item def DynamicsConstraint {
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute mass : Real;
        attribute appliedForce : Real;
        attribute time : Real;
    }
}