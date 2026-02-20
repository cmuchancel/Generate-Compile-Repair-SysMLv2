package AutomotiveComponentModel {
    private import ScalarValues::*;

    enum def LengthUnit {
        enum millimeter;
        enum centimeter;
        enum meter;
        enum inch;
        enum foot;
    }

    enum def MassUnit {
        enum gram;
        enum kilogram;
        enum pound;
    }

    part def Vehicle {
        attribute mass : ScalarValues::Real;
        attribute massUnit : MassUnit = MassUnit::kilogram;
        attribute length : ScalarValues::Real;
        attribute lengthUnit : LengthUnit = LengthUnit::meter;
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }

    part def Wheel {
        attribute hubDiameter : ScalarValues::Real;
        attribute hubDiameterUnit : LengthUnit = LengthUnit::inch;
        attribute width : ScalarValues::Real;
        attribute widthUnit : LengthUnit = LengthUnit::millimeter;
        attribute outerDiameter : ScalarValues::Real;
        attribute outerDiameterUnit : LengthUnit = LengthUnit::inch;
        part tire : Tire;
    }

    part def Tire {
        attribute height : ScalarValues::Real;
        attribute heightUnit : LengthUnit = LengthUnit::millimeter;
    }

    part vehicle : Vehicle {
        attribute mass : ScalarValues::Real = 1200;
        attribute length : ScalarValues::Real = 4.82;
        part leftFrontWheel : Wheel {
            attribute hubDiameter : ScalarValues::Real = 18;
            attribute width : ScalarValues::Real = 245;
        }
        part rightFrontWheel : Wheel {
            attribute hubDiameter : ScalarValues::Real = 18;
            attribute width : ScalarValues::Real = 245;
        }
    }
}