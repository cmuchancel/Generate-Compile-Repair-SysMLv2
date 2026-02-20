package WheelAssemblyModel {
  private import ScalarValues::*;

  part def WheelBearingUnit { }
  part def Bead { }
  part def Tire {
    part bead1 : Bead;
    part bead2 : Bead;
  }
  part def Bolt { }
  part def BoltMountingHole { }
  part def ThreadedHole { }
  part def Rim {
    part boltMountingHole1 : BoltMountingHole;
    part boltMountingHole2 : BoltMountingHole;
  }
  part def Hub {
    part threadedHole1 : ThreadedHole;
    part threadedHole2 : ThreadedHole;
  }

  part def WheelAssembly {
    part bearingUnit : WheelBearingUnit;
    part hub : Hub;
    part rimA : Rim;
    part rimB : Rim;
    part tire : Tire;
    part bolt1 : Bolt;
    part bolt2 : Bolt;

    connect bearingUnit to hub;
    connect tire.bead1 to rimA;
    connect tire.bead2 to rimB;
    connect rimA.boltMountingHole1 to bolt1;
    connect hub.threadedHole1 to bolt1;
    connect rimA.boltMountingHole2 to bolt2;
    connect hub.threadedHole2 to bolt2;
  }

  part wheelAssembly : WheelAssembly;
}