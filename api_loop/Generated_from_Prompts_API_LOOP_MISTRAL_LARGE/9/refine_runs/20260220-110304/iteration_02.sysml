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
        part bead1 : Bead;
        part bead2 : Bead;
        part hub : Hub;
        part rim1 : Rim;
        part rim2 : Rim;

        attribute numMountingHoles : Integer = 5;
        part mountingHoles : BoltMountingHole[numMountingHoles];
        part threadedHoles : ThreadedHole[numMountingHoles];
        part bolts : Bolt[numMountingHoles];

        port pressFit1 : PressFitPort;
        port pressFit2 : PressFitPort;
        port boltPorts : BoltConnectionPort[numMountingHoles];
    }

    connect wheelAssembly.bead1 to wheelAssembly.rim1;
    connect wheelAssembly.bead2 to wheelAssembly.rim2;

    connect wheelAssembly.mountingHoles[0] to wheelAssembly.threadedHoles[0] via wheelAssembly.bolts[0];
    connect wheelAssembly.mountingHoles[1] to wheelAssembly.threadedHoles[1] via wheelAssembly.bolts[1];
    connect wheelAssembly.mountingHoles[2] to wheelAssembly.threadedHoles[2] via wheelAssembly.bolts[2];
    connect wheelAssembly.mountingHoles[3] to wheelAssembly.threadedHoles[3] via wheelAssembly.bolts[3];
    connect wheelAssembly.mountingHoles[4] to wheelAssembly.threadedHoles[4] via wheelAssembly.bolts[4];

    connect wheelAssembly.boltPorts[0] to wheelAssembly.bolts[0];
    connect wheelAssembly.boltPorts[1] to wheelAssembly.bolts[1];
    connect wheelAssembly.boltPorts[2] to wheelAssembly.bolts[2];
    connect wheelAssembly.boltPorts[3] to wheelAssembly.bolts[3];
    connect wheelAssembly.boltPorts[4] to wheelAssembly.bolts[4];
}