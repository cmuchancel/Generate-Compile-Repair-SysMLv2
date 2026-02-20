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

    connect wheelAssembly.bead1 to wheelAssembly.rim1 via wheelAssembly.pressFit1;
    connect wheelAssembly.bead2 to wheelAssembly.rim2 via wheelAssembly.pressFit2;

    for i in 1..wheelAssembly.numMountingHoles {
        connect wheelAssembly.mountingHoles[i] to wheelAssembly.threadedHoles[i] via wheelAssembly.bolts[i];
        connect wheelAssembly.boltPorts[i] to wheelAssembly.bolts[i];
    }
}