package VehicleMassModel {
    private import ScalarValues::*;

    part def Component {
        attribute mass : Real;
    }

    part def Engine :> Component;
    part def Transmission :> Component;

    part def Vehicle {
        part engine : Engine;
        part transmission : Transmission;
        attribute vehicleMass : Real;
    }
}