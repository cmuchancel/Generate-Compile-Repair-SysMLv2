package VehiclePlatform {
    private import ScalarValues::*;

    port def VehiclePort;

    part def part1;
    part def part2;
    part def part3 {
        port p1 : VehiclePort;
    }
    part def part4;
    part def part5 {
        port p2 : VehiclePort;
        attribute behaviorVariant : Integer = 0;
    }
    part def part6;

    part def SubsystemA {
        attribute variantID : Integer = 0;
        part p1_inst : part1;
        part p2_inst : part2;
        part p3_inst : part3;
        part p4_inst : part4;
        port subsystemPort : VehiclePort;
        connect p3_inst.p1 to subsystemPort;
    }

    part def SubsystemB {
        attribute variantID : Integer = 0;
        part p5_inst : part5;
        part p6_inst : part6;
        port subsystemPort : VehiclePort;
        connect p5_inst.p2 to subsystemPort;
    }

    part def VehicleConfigA {
        part subA : SubsystemA;
        part subB : SubsystemB;
        connect subA.subsystemPort to subB.subsystemPort;
        attribute configurationParam : Real = 1.0;
    }

    part def VehicleConfigB {
        part subA : SubsystemA;
        part subB : SubsystemB;
        connect subA.subsystemPort to subB.subsystemPort;
        attribute configurationParam : Real = 2.0;
    }

    item def SystemLevelConstraint {
        attribute maxConnections : Integer = 10;
        attribute minConnections : Integer = 1;
    }
}