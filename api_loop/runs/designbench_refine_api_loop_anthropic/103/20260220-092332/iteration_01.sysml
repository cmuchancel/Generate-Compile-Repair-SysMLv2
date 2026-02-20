package MultiArchSystem {
    private import ScalarValues::*;

    port def SignalPort;

    part def C1 {
        port pb : SignalPort;
    }

    part def C2 {
        port pc : SignalPort;
    }

    part def C3 {
        port pd : SignalPort;
    }

    part def C4 {
        port pe_internal : SignalPort;
    }

    port def InterfacePE;
    port def InterfacePF;

    part def A11 {
        part c1 : C1;
        part c2 : C2;
        part c3 : C3;
        part c4 : C4;
    }

    part def B11 {
        part c1 : C1;
        part c2 : C2;
        port pe : InterfacePE;
        connect c1.pb to c2.pc;
    }

    part def B12 {
        part c3 : C3;
        part c4 : C4;
        port pf : InterfacePF;
    }

    part def System1 {
        part b11 : B11;
        part b12 : B12;
        connect b11.pe to b12.pf;
    }

    part def B21 {
        part c1 : C1;
        part c2 : C2;
    }

    part def B22 {
        part c3 : C3;
        part c4 : C4;
    }

    part def System2 {
        part b21 : B21;
        part b22 : B22;
        connect b21.c1.pb to b21.c2.pc;
    }

    part topSystem : System1;
    part topSystem2 : System2;
}