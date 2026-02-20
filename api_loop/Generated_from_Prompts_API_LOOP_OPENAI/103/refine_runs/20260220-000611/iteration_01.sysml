package ArchitectureModel {
    private import ScalarValues::*;

    item def Signal;

    port def SigPort {
        item sig : Signal;
    }

    part def C1 {
        port pb : SigPort;
    }

    part def C2 {
        port pc : SigPort;
    }

    part def C3 {
        port pd : SigPort;
    }

    part def C4 {
    }

    part def B11 {
        part c1 : C1;
        part c2 : C2;
        port pe : SigPort;

        connect c1.pb to c2.pc;
        connect pe to c1.pb;
    }

    part def B12 {
        part c3 : C3;
        part c4 : C4;
        port pf : SigPort;

        connect pf to c3.pd;
    }

    part def A11 {
        part b11 : B11;
        part b12 : B12;

        connect b11.pe to b12.pf;
    }

    part def B21 {
        part c1 : C1;
        part c2 : C2;
        port p21 : SigPort;

        connect c1.pb to c2.pc;
        connect p21 to c1.pb;
    }

    part def B22 {
        part c3 : C3;
        part c4 : C4;
        port p22 : SigPort;

        connect p22 to c3.pd;
    }

    part def A21 {
        part b21 : B21;
        part b22 : B22;

        connect b21.p21 to b22.p22;
    }

    part def System {
        part a11 : A11;
        part a21 : A21;
    }

    part system : System;
}