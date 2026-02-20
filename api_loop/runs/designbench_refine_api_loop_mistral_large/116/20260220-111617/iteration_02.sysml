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
        attribute variant : ComponentVariant;
        attribute variant_value : String = "ComponentVariant::basic";
    }
    part def Part6;

    part def SubsystemA {
        part part1 : Part1;
        part part2 : Part2;
        part part3 : Part3;
        part part5 : Part5;
    }

    part def SubsystemB {
        part part4 : Part4;
        part part5 : Part5;
        part part6 : Part6;
    }

    part def VehicleConfigA {
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;

        connect subsystemA.part3.p1 to subsystemB.part5.p2;
    }

    part def VehicleConfigB {
        part subsystemA : SubsystemA;
        part subsystemB : SubsystemB;

        connect subsystemA.part5.p2 to subsystemB.part5.p2;
    }
}