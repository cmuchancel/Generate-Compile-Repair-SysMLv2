package VehicleStructure {
    part def VehicleComponent;
    part def Engine : VehicleComponent;
    part def Transmission : VehicleComponent;
    part def Wheel : VehicleComponent;

    part def Vehicle {
        part engine : Engine;
        part transmission : Transmission;
        part wheels[4] : Wheel;
    }

    part def Car : Vehicle;
}