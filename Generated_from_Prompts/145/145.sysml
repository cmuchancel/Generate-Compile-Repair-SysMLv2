package 'TireInformationSystem' {
	private import ScalarValues::*;
	private import SI::*;
	private import USCustomaryUnits::*;
	item def PositionVector {
		attribute x : Real;
		attribute y : Real;
		attribute z : Real;
	}
	part def Tire {
		attribute manufacturer : String;
		attribute rimDiameter : LengthValue;
		attribute width : Integer;
		part installationPosition : PositionVector[0..1];
	}
	part def MichelinPerformanceTire :> Tire {
		attribute redefines manufacturer = "Michelin";
		attribute redefines rimDiameter = 18 ['in'];
		attribute redefines width = 245;
	}
	part def TireCatalog {
		part definedTires : Tire[*];
		part michelinExample : MichelinPerformanceTire;
	}
}
