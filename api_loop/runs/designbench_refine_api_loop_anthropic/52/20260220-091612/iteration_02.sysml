package VehicleSystem {
    private import ScalarValues::*;

    enum def GearSetting {
        enum GEAR_0;
        enum GEAR_1;
        enum GEAR_2;
        enum GEAR_3;
        enum GEAR_4;
        enum GEAR_5;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute gear : GearSetting;
        attribute acceleratorPosition : Real;
    }

    part def Engine;

    part def Driver;

    part def CompositeSystem {
        attribute vehicleMass : Real;
        attribute vehicleGear : GearSetting;
        attribute vehicleAcceleratorPosition : Real;
    }

    part Vehicle_1_t0 : Vehicle {
        attribute redefines mass = 2000.0;
        attribute redefines gear = GearSetting::GEAR_0;
        attribute redefines acceleratorPosition = 0.0;
    }

    part Vehicle_1_t1 : Vehicle {
        attribute redefines mass = 1500.0;
        attribute redefines gear = GearSetting::GEAR_2;
        attribute redefines acceleratorPosition = 0.5;
    }
}