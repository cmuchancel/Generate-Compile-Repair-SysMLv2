package AutomobileWheelAssembly {
    private import ScalarValues::*;

    part def WheelBearingUnit;
    part def Tire;
    part def Bead;
    part def Hub;
    part def Rim;
    part def Bolt;
    part def BoltMountingHole;
    part def ThreadedHole;

    part def WheelAssembly {
        part bearingUnit : WheelBearingUnit;
        part tire : Tire {
            part bead1 : Bead;
            part bead2 : Bead;
        };
        part hub : Hub {
            part rim1 : Rim;
            part rim2 : Rim;
            part mountingHoles : BoltMountingHole[0..*];
            part threadedHoles : ThreadedHole[0..*];
        };
        part bolts : Bolt[0..*];
    };

    connect WheelAssembly.tire.bead1 to WheelAssembly.hub.rim1;
    connect WheelAssembly.tire.bead2 to WheelAssembly.hub.rim2;

    for i in 1..WheelAssembly.hub.mountingHoles->size() {
        connect WheelAssembly.hub.mountingHoles[i] to WheelAssembly.hub.threadedHoles[i] using WheelAssembly.bolts[i];
    };
}