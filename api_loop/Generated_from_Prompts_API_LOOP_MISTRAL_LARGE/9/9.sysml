package WheelAssembly {
    private import ScalarValues::*;

    part def WheelBearingUnit;
    part def Tire;
    part def Bead;
    part def Hub;
    part def Rim;
    part def Bolt;
    part def BoltMountingHole;
    part def ThreadedHole;

    part wheelAssembly {
        part bearingUnit : WheelBearingUnit;
        part tire : Tire {
            part bead1 : Bead;
            part bead2 : Bead;
        }
        part hub : Hub {
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
        part bolt1 : Bolt;
        part bolt2 : Bolt;
        part bolt3 : Bolt;
        part bolt4 : Bolt;
    }

    connect wheelAssembly.tire.bead1 to wheelAssembly.hub.rim1;
    connect wheelAssembly.tire.bead2 to wheelAssembly.hub.rim2;

    connect wheelAssembly.hub.mountingHole1 to wheelAssembly.bolt1;
    connect wheelAssembly.bolt1 to wheelAssembly.hub.threadedHole1;

    connect wheelAssembly.hub.mountingHole2 to wheelAssembly.bolt2;
    connect wheelAssembly.bolt2 to wheelAssembly.hub.threadedHole2;

    connect wheelAssembly.hub.mountingHole3 to wheelAssembly.bolt3;
    connect wheelAssembly.bolt3 to wheelAssembly.hub.threadedHole3;

    connect wheelAssembly.hub.mountingHole4 to wheelAssembly.bolt4;
    connect wheelAssembly.bolt4 to wheelAssembly.hub.threadedHole4;
}