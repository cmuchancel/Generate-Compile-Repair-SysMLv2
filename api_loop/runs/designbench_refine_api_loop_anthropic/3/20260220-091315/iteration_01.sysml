package VehicleStructure {
    private import ScalarValues::*;

    enum def ComponentType {
        enum ENGINE;
        enum TRANSMISSION;
        enum WHEEL;
    }

    part def VehicleComponent {
        attribute componentId : String;
        attribute componentName : String;
    }

    part def Engine :> VehicleComponent {
        attribute displacement : Real;
        attribute horsepower : Real;
    }

    part def Transmission :> VehicleComponent {
        attribute gearCount : Integer;
        attribute transmissionType : String;
    }

    part def Wheel :> VehicleComponent {
        attribute diameter : Real;
        attribute width : Real;
    }

    part def Vehicle {
        attribute vehicleId : String;
        attribute vehicleName : String;
        attribute vehicleModel : String;

        part engine : Engine[1];
        part transmission : Transmission[1];
        part wheels : Wheel[4];
    }

    part def StandardVehicle :> Vehicle {
        attribute manufacturer : String;
        attribute year : Integer;
    }

    part def ElectricVehicle :> Vehicle {
        attribute batteryCapacity : Real;
        attribute range : Real;
    }

    part def PerformanceEngine :> Engine {
        attribute turboCharged : Boolean;
        attribute maxRPM : Integer;
    }

    part def AutomaticTransmission :> Transmission {
        attribute hasPaddleShifters : Boolean;
    }

    part def ManualTransmission :> Transmission {
        attribute clutchType : String;
    }

    part def AlloyWheel :> Wheel {
        attribute material : String;
        attribute finish : String;
    }

    part myVehicle : StandardVehicle {
        attribute vehicleId = "V001";
        attribute vehicleName = "Sedan";
        attribute vehicleModel = "Model-X";
        attribute manufacturer = "AutoCorp";
        attribute year = 2024;

        part engine : PerformanceEngine[1] {
            attribute componentId = "E001";
            attribute componentName = "V8 Engine";
            attribute displacement = 5.0;
            attribute horsepower = 450.0;
            attribute turboCharged = true;
            attribute maxRPM = 7000;
        }

        part transmission : AutomaticTransmission[1] {
            attribute componentId = "T001";
            attribute componentName = "8-Speed Auto";
            attribute gearCount = 8;
            attribute transmissionType = "Automatic";
            attribute hasPaddleShifters = true;
        }

        part wheels : AlloyWheel[4] {
            attribute componentId = "W001";
            attribute componentName = "Sport Alloy";
            attribute diameter = 19.0;
            attribute width = 8.5;
            attribute material = "Aluminum";
            attribute finish = "Gloss Black";
        }
    }
}