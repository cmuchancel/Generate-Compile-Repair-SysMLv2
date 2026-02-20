package modular_vehicle_platform {
    private import ScalarValues::*;

    enum def ComponentVariant {
        enum basic;
        enum extended;
        enum premium;
    }

    part def Part1;
    part def Part2;
    part def Part3 {
        port p1;
    }
    part def Part4;
    part def Part5 {
        port p2;
        attribute variantValue : ComponentVariant = ComponentVariant::basic;
    }
    part def Part6;

    part def SubsystemA {
        part part1 : Part1;
        part part2 : Part2;
        part part3 : Part3;
        part part4 : Part4;
        attribute componentCount : Integer = 4;
    }

    part def SubsystemB {
        part part5 : Part5;
        part part6 : Part6;
        attribute componentCount : Integer = 2;
    }

    part def VehicleConfigA {
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;

        connect subsystemA.part3.p1 to subsystemB.part5.p2;
    }

    part def VehicleConfigB {
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;

        connect subsystemA.part3.p1 to subsystemB.part5.p2;
        attribute configVariant : ComponentVariant = ComponentVariant::extended;
    }
}