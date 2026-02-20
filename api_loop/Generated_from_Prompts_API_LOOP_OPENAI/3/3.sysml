package VehicleStructure {
    part def VehicleComponent;
    part def Engine specializes VehicleComponent;
    part def Transmission specializes VehicleComponent;
    part def Wheel specializes VehicleComponent;

    part def Vehicle {
        part engine : Engine;
        part transmission : Transmission;
        part wheels[4] : Wheel;
    }

    part def Car specializes Vehicle;
}