package 'ConfigurableEngineVariants' {
	private import SI::*;
	abstract part def ConfigurableEngine {
		part cylinders : Cylinder[4..6];
	}
	part def FourCylinderEngine :> ConfigurableEngine {
		part redefines cylinders[4];
	}
	part def SixCylinderEngine :> ConfigurableEngine {
		part redefines cylinders[6];
	}
	abstract part def Cylinder {
		attribute diameter : LengthValue;
	}
	part def SmallDiameterCylinder :> Cylinder {
		attribute redefines diameter = 70 [mm];
	}
	part def LargeDiameterCylinder :> Cylinder {
		attribute redefines diameter = 100 [mm];
	}
}
