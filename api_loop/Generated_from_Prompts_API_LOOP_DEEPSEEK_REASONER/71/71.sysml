package VehicleSystem {
    private import ScalarValues::*;
    
    part def Vehicle {
        attribute curbMass : Real;
        attribute fuelMass : Real;
        attribute totalMass : Real;
    }
    
    part def FuelTank {
        attribute capacity : Real;
        attribute currentLevel : Real;
    }
    
    part def Engine {
        attribute torque : Real;
        attribute speed : Real;
    }
    
    part def Clutch {
        attribute engagement : Real;
    }
    
    part def Gearbox {
        attribute gearRatio : Real;
    }
    
    port def TorquePort;
    
    port def MechanicalInterface;
    
    part myVehicle : Vehicle {
        attribute curbMass : Real = 1400;
        attribute fuelMass : Real;
        attribute totalMass : Real;
        
        part fuelTank : FuelTank {
            attribute capacity : Real = 100;
            attribute currentLevel : Real = 100;
        }
        
        part engine : Engine {
            attribute torque : Real;
            attribute speed : Real;
            port torqueOutput : TorquePort;
        }
        
        part clutch : Clutch {
            attribute engagement : Real = 1.0;
            port input : MechanicalInterface;
            port output : MechanicalInterface;
        }
        
        part gearbox : Gearbox {
            attribute gearRatio : Real = 3.5;
            port input : MechanicalInterface;
        }
        
        connect engine.torqueOutput to clutch.input;
        connect clutch.output to gearbox.input;
    }
}