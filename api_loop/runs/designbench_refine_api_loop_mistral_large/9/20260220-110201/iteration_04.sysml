package AutomobileWheelAssembly {
    private import ScalarValues::*;

    part def WheelBearingUnit;
    part def Tire;
    part def Bead;
    part def Hub;
    part def Rim;
    part def BoltMountingHole;
    part def ThreadedHole;
    part def Bolt;

    part wheelAssembly {
        part bearingUnit : WheelBearingUnit;
        part tire : Tire {
            part bead1 : Bead;
            part bead2 : Bead;
        }
        part hub : Hub {
            part rim1 : Rim;
            part rim2 : Rim;
            attribute mountingHoleCount : Integer = 5;
            part mountingHoles : BoltMountingHole[mountingHoleCount];
            part threadedHoles : ThreadedHole[mountingHoleCount];
        }
        part bolts : Bolt[hub.mountingHoleCount];
    }

    connect wheelAssembly.tire.bead1 to wheelAssembly.hub.rim1;
    connect wheelAssembly.tire.bead2 to wheelAssembly.hub.rim2;

    connect wheelAssembly.hub.mountingHoles[1] to wheelAssembly.bolts[1];
    connect wheelAssembly.bolts[1] to wheelAssembly.hub.threadedHoles[1];
    connect wheelAssembly.hub.mountingHoles[2] to wheelAssembly.bolts[2];
    connect wheelAssembly.bolts[2] to wheelAssembly.hub.threadedHoles[2];
    connect wheelAssembly.hub.mountingHoles[3] to wheelAssembly.bolts[3];
    connect wheelAssembly.bolts[3] to wheelAssembly.hub.threadedHoles[3];
    connect wheelAssembly.hub.mountingHoles[4] to wheelAssembly.bolts[4];
    connect wheelAssembly.bolts[4] to wheelAssembly.hub.threadedHoles[4];
    connect wheelAssembly.hub.mountingHoles[5] to wheelAssembly.bolts[5];
    connect wheelAssembly.bolts[5] to wheelAssembly.hub.threadedHoles[5];
}