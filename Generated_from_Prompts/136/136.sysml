package 'TireSpecificationModel' {
	private import ScalarValues::*;
	part def Tire {
		attribute manufacturer : String;
		attribute rimDiameter : Real;
		attribute width : Integer;
	}
	part michelinTire : Tire {
		attribute manufacturer = "Michelin";
		attribute rimDiameter = 18.0;
		attribute width = 245;
	}
}
