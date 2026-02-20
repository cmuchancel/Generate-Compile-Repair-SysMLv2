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

    part def VehicleState {
        attribute gear : GearSetting;
        attribute acceleratorPosition : Real;
    }

    part def Vehicle {
        attribute mass : Real;
        part state : VehicleState;
    }

    part def Engine;

    part def Driver;

    part def CompositeSystem {
        part vehicle : Vehicle;
        part engine : Engine;
        part driver : Driver;
    }

    part def VehicleSnapshot {
        attribute mass : Real;
        attribute gear : GearSetting;
        attribute acceleratorPosition : Real;
    }

    part Vehicle_1_t0 : VehicleSnapshot {
        attribute redefines mass = 2000.0;
        attribute redefines gear = GearSetting::GEAR_0;
        attribute redefines acceleratorPosition = 0.0;
    }

    part Vehicle_1_t1 : VehicleSnapshot {
        attribute redefines mass = 1500.0;
        attribute redefines gear = GearSetting::GEAR_2;
        attribute redefines acceleratorPosition = 0.5;
    }

    part def Vehicle1System {
        part initialState : VehicleSnapshot {
            attribute redefines mass = 2000.0;
            attribute redefines gear = GearSetting::GEAR_0;
            attribute redefines acceleratorPosition = 0.0;
        }
        part nextState : VehicleSnapshot {
            attribute redefines mass = 1500.0;
            attribute redefines gear = GearSetting::GEAR_2;
            attribute redefines acceleratorPosition = 0.5;
        }
    }
}