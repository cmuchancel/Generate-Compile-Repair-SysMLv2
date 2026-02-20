package FuelSystem {
    private import ScalarValues::*;

    port def FuelInterface;
    
    part def FuelTank {
        port fuelSupply : FuelInterface;
        port fuelReturn : FuelInterface;
        attribute temperature : Real;
    }
    
    part def Engine {
        port fuelIn : FuelInterface;
        port fuelOut : FuelInterface;
    }
    
    part def FuelManagementSystem {
        part tank : FuelTank;
        part engine : Engine;
        
        connect tank.fuelSupply to engine.fuelIn;
        connect engine.fuelOut to tank.fuelReturn;
    }
}