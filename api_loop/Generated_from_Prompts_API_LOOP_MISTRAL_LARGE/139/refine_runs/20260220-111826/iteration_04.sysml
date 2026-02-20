package AutomotiveSystem {
    private import ScalarValues::*;

    attribute def MassValue : Real;
    attribute def LengthValue : Real;

    part def Car {
        attribute totalMass : MassValue = MassValue(1200.0);
        attribute length : LengthValue = LengthValue(4.82);
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
        attribute diameter : LengthValue = LengthValue(18.0);
    }

    part def Tire {
        attribute width : LengthValue = LengthValue(245.0);
        attribute height : LengthValue = LengthValue(45.0);
        attribute treadDepth : LengthValue = LengthValue(6.0);
        attribute minTreadDepth : LengthValue = LengthValue(3.5);
        attribute outerDiameter : LengthValue;
    }

    part def BrakeDisc {
        attribute radius : LengthValue = LengthValue(95.0);
        attribute diameter : LengthValue = LengthValue(2 * radius.value);
    }

    part car : Car;

    constraint def WheelAssemblyFitment {
        for wheelAssembly in car.wheelAssemblies {
            wheelAssembly.brakeDisc.diameter.value < wheelAssembly.wheel.tire.outerDiameter.value
        }
    }

    constraint def TreadDepthCompliance {
        for wheelAssembly in car.wheelAssemblies {
            wheelAssembly.wheel.tire.treadDepth.value >= wheelAssembly.wheel.tire.minTreadDepth.value
        }
    }

    attribute def CalculateOuterDiameter : LengthValue {
        value = (car.wheelAssemblies[1].wheel.rim.diameter.value * 25.4) + (2 * car.wheelAssemblies[1].wheel.tire.height.value)
    }

    for wheelAssembly in car.wheelAssemblies {
        wheelAssembly.wheel.tire.outerDiameter = CalculateOuterDiameter;
    }
}