package AutomotiveSystem {
    private import ScalarValues::*;

    attribute def MassValue {
        value : Real;
    }

    attribute def LengthValue {
        value : Real;
    }

    part def Car {
        attribute totalMass : MassValue = (value => 1200.0);
        attribute length : LengthValue = (value => 4.82);
        part wheelAssemblies : WheelAssembly[4];
    }

    part def WheelAssembly {
        part wheel : Wheel;
        part brakeDisc : BrakeDisc;
    }

    part def Wheel {
        part rim : Rim;
        part tire : Tire;
    }

    part def Rim {
        attribute diameter : LengthValue = (value => 18.0);
    }

    part def Tire {
        attribute width : LengthValue = (value => 245.0);
        attribute height : LengthValue = (value => 45.0);
        attribute treadDepth : LengthValue = (value => 6.0);
        attribute minTreadDepth : LengthValue = (value => 3.5);
        attribute outerDiameter : LengthValue;
    }

    part def BrakeDisc {
        attribute radius : LengthValue = (value => 95.0);
        attribute diameter : LengthValue = (value => 2 * radius.value);
    }

    part car : Car;

    constraint def WheelAssemblyFitment {
        car.wheelAssemblies->forAll(wa |
            wa.brakeDisc.diameter.value < wa.wheel.tire.outerDiameter.value
        )
    }

    constraint def TreadDepthCompliance {
        car.wheelAssemblies->forAll(wa |
            wa.wheel.tire.treadDepth.value >= wa.wheel.tire.minTreadDepth.value
        )
    }

    attribute def CalculateOuterDiameter : LengthValue {
        value = (car.wheelAssemblies[1].wheel.rim.diameter.value * 25.4) + (2 * car.wheelAssemblies[1].wheel.tire.height.value);
    }

    car.wheelAssemblies->forAll(wa |
        wa.wheel.tire.outerDiameter = CalculateOuterDiameter
    )
}