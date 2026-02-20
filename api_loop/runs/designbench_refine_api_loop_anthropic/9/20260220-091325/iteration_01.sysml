package WheelAssembly {
    private import ScalarValues::*;

    enum def ConnectionType {
        enum Press;
        enum Bolt;
        enum Interference;
    }

    part def WheelBearingUnit {
        attribute description : String;
    }

    part def Bead {
        attribute position : String;
    }

    part def Rim {
        attribute position : String;
    }

    part def Tire {
        attribute sealed : Boolean;
        part bead1 : Bead;
        part bead2 : Bead;
    }

    part def BoltMountingHole {
        attribute diameter : Real;
    }

    part def ThreadedHole {
        attribute diameter : Real;
    }

    part def BoltConnection {
        attribute torque : Real;
    }

    part def Hub {
        part rim1 : Rim;
        part rim2 : Rim;
        part mountingHole1 : BoltMountingHole;
        part mountingHole2 : BoltMountingHole;
        part mountingHole3 : BoltMountingHole;
        part mountingHole4 : BoltMountingHole;
        part threadedHole1 : ThreadedHole;
        part threadedHole2 : ThreadedHole;
        part threadedHole3 : ThreadedHole;
        part threadedHole4 : ThreadedHole;
    }

    part def WheelAssemblySystem {
        attribute description : String;

        part bearing : WheelBearingUnit {
            attribute description : String = "Supports wheel rotation";
        }

        part hub : Hub;

        part tire : Tire {
            attribute sealed : Boolean = true;
        }

        part bolt1 : BoltConnection {
            attribute torque : Real = 120.0;
        }

        part bolt2 : BoltConnection {
            attribute torque : Real = 120.0;
        }

        part bolt3 : BoltConnection {
            attribute torque : Real = 120.0;
        }

        part bolt4 : BoltConnection {
            attribute torque : Real = 120.0;
        }

        connect tire.bead1 to hub.rim1;
        connect tire.bead2 to hub.rim2;
        connect bolt1 to hub.mountingHole1;
        connect bolt1 to hub.threadedHole1;
        connect bolt2 to hub.mountingHole2;
        connect bolt2 to hub.threadedHole2;
        connect bolt3 to hub.mountingHole3;
        connect bolt3 to hub.threadedHole3;
        connect bolt4 to hub.mountingHole4;
        connect bolt4 to hub.threadedHole4;
        connect hub to bearing;
    }
}