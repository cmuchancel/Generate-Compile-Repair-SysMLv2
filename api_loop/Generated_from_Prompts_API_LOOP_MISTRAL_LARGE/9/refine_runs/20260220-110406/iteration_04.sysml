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

    port def PressFitPort;
    port def BoltConnectionPort;

    part wheelAssembly {
        part bearingUnit : WheelBearingUnit;
        part tire : Tire;
        part hub : Hub;

        part rim1 : Rim;
        part rim2 : Rim;

        part bead1 : Bead;
        part bead2 : Bead;

        part mountingHoles : BoltMountingHole[4];
        part threadedHoles : ThreadedHole[4];
        part bolts : Bolt[4];

        port pressFit1 : PressFitPort;
        port pressFit2 : PressFitPort;
        port boltPorts : BoltConnectionPort[4];
    }

    connect wheelAssembly.bead1 to wheelAssembly.rim1 via pressFit1;
    connect wheelAssembly.bead2 to wheelAssembly.rim2 via pressFit2;

    for i in 1..4 {
        connect wheelAssembly.mountingHoles[i] to wheelAssembly.threadedHoles[i];
        connect wheelAssembly.bolts[i] to wheelAssembly.mountingHoles[i];
        connect wheelAssembly.bolts[i] to wheelAssembly.boltPorts[i];
    }
}