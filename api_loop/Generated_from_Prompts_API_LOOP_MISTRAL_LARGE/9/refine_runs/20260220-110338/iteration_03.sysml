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

    connect wheelAssembly.leftBead to wheelAssembly.leftRim;
    connect wheelAssembly.rightBead to wheelAssembly.rightRim;

    for i in 1..wheelAssembly.mountingHoleCount {
        connect wheelAssembly.mountingHoles[i] to wheelAssembly.bolts[i];
        connect wheelAssembly.bolts[i] to wheelAssembly.threadedHoles[i];
    }
}