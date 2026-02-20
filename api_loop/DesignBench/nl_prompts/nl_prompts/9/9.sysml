package 'WheelAssemblyStructure' {
	private import ScalarValues::*;

	part def WheelAssembly {
		part bearingUnit : WheelBearingUnit;
		part hub : Hub;
		part tire : Tire;
		part boltConnection : BoltConnection[0..*];
	}

	part def WheelBearingUnit;

	part def Hub {
		part rim : Rim[2];
		part mountingHole : BoltMountingHole[0..*];
		part threadedHole : ThreadedHole[0..*];
	}

	part def Tire {
		part bead : Bead[2];
		ref part seatedRim : Rim[2];
	}

	part def Bead;
	part def Rim;

	part def BoltMountingHole;
	part def ThreadedHole;

	part def BoltConnection {
		ref part mountingHole : BoltMountingHole;
		ref part threadedHole : ThreadedHole;
		part bolt : Bolt;
	}

	part def Bolt;
}
