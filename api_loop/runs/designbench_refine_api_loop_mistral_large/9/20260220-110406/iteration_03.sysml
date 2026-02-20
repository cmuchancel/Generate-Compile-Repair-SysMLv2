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

    connect wheelAssembly.bead1 to wheelAssembly.rim1;
    connect wheelAssembly.bead2 to wheelAssembly.rim2;

    connect wheelAssembly.mountingHoles[1] to wheelAssembly.threadedHoles[1];
    connect wheelAssembly.bolts[1] to wheelAssembly.mountingHoles[1];
    connect wheelAssembly.bolts[1] to wheelAssembly.boltPorts[1];

    connect wheelAssembly.mountingHoles[2] to wheelAssembly.threadedHoles[2];
    connect wheelAssembly.bolts[2] to wheelAssembly.mountingHoles[2];
    connect wheelAssembly.bolts[2] to wheelAssembly.boltPorts[2];

    connect wheelAssembly.mountingHoles[3] to wheelAssembly.threadedHoles[3];
    connect wheelAssembly.bolts[3] to wheelAssembly.mountingHoles[3];
    connect wheelAssembly.bolts[3] to wheelAssembly.boltPorts[3];

    connect wheelAssembly.mountingHoles[4] to wheelAssembly.threadedHoles[4];
    connect wheelAssembly.bolts[4] to wheelAssembly.mountingHoles[4];
    connect wheelAssembly.bolts[4] to wheelAssembly.boltPorts[4];
}