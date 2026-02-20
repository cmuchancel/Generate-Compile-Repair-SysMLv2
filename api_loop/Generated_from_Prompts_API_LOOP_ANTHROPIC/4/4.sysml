package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleSize {
        enum Small;
        enum Large;
    }

    part def Cylinder;

    part def SmallEngine {
        part cylinder1 : Cylinder;
        part cylinder2 : Cylinder;
        part cylinder3 : Cylinder;
        part cylinder4 : Cylinder;
    }

    part def LargeEngine {
        part cylinder1 : Cylinder;
        part cylinder2 : Cylinder;
        part cylinder3 : Cylinder;
        part cylinder4 : Cylinder;
        part cylinder5 : Cylinder;
        part cylinder6 : Cylinder;
    }

    part def SmallVehicle {
        part engine : SmallEngine;
    }

    part def LargeVehicle {
        part engine : LargeEngine;
    }

    part smallVehicleInstance : SmallVehicle {
        part engine : SmallEngine;
    }

    part largeVehicleInstance : LargeVehicle {
        part engine : LargeEngine;
    }
}