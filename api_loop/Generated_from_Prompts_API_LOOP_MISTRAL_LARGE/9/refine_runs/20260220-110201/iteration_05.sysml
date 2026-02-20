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

    for i in 1..wheelAssembly.hub.mountingHoleCount {
        connect wheelAssembly.hub.mountingHoles[i] to wheelAssembly.bolts[i];
        connect wheelAssembly.bolts[i] to wheelAssembly.hub.threadedHoles[i];
    }
}