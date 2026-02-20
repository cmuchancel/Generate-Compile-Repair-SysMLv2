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
            part mountingHoles : BoltMountingHole[4];
            part threadedHoles : ThreadedHole[4];
        }
        part bolts : Bolt[4];
    }

    connect wheelAssembly.tire.bead1 to wheelAssembly.hub.rim1;
    connect wheelAssembly.tire.bead2 to wheelAssembly.hub.rim2;

    for i in 1..4 {
        connect wheelAssembly.hub.mountingHoles[i] to wheelAssembly.hub.threadedHoles[i] using wheelAssembly.bolts[i];
    }
}