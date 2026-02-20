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
        part tire : Tire;
        part hub : Hub;

        part leftBead : Bead;
        part rightBead : Bead;
        part leftRim : Rim;
        part rightRim : Rim;

        attribute mountingHoleCount : Integer = 5;
        part mountingHoles : BoltMountingHole[mountingHoleCount];
        part threadedHoles : ThreadedHole[mountingHoleCount];
        part bolts : Bolt[mountingHoleCount];
    }

    connect wheelAssembly.leftBead, wheelAssembly.leftRim;
    connect wheelAssembly.rightBead, wheelAssembly.rightRim;

    connect wheelAssembly.mountingHoles[1], wheelAssembly.bolts[1];
    connect wheelAssembly.bolts[1], wheelAssembly.threadedHoles[1];
    connect wheelAssembly.mountingHoles[2], wheelAssembly.bolts[2];
    connect wheelAssembly.bolts[2], wheelAssembly.threadedHoles[2];
    connect wheelAssembly.mountingHoles[3], wheelAssembly.bolts[3];
    connect wheelAssembly.bolts[3], wheelAssembly.threadedHoles[3];
    connect wheelAssembly.mountingHoles[4], wheelAssembly.bolts[4];
    connect wheelAssembly.bolts[4], wheelAssembly.threadedHoles[4];
    connect wheelAssembly.mountingHoles[5], wheelAssembly.bolts[5];
    connect wheelAssembly.bolts[5], wheelAssembly.threadedHoles[5];
}