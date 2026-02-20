package VehicleStructure {
    private import ScalarValues::*;

    part def VehicleComponent;

    part def Engine :> VehicleComponent;
    
    part def Transmission :> VehicleComponent;
    
    part def Wheel :> VehicleComponent;
    
    part def Vehicle {
        part engine : Engine[1];
        part transmission : Transmission[1];
        part wheels : Wheel[4];
    }
}