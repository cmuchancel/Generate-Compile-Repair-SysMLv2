package MultiArchitectureSystem {
    private import ScalarValues::*;
    
    port def SignalPort;
    
    part def c1 {
        port pb : SignalPort;
    }
    
    part def c2 {
        port pc : SignalPort;
    }
    
    part def c3 {
        port pd : SignalPort;
    }
    
    part def c4 {
        port pg : SignalPort;
    }
    
    part def b11 {
        port pe : SignalPort;
        part comp1 : c1;
        part comp2 : c2;
        connect pe to comp1.pb;
        connect comp1.pb to comp2.pc;
    }
    
    part def b12 {
        port pf : SignalPort;
        part comp3 : c3;
        part comp4 : c4;
        connect pf to comp3.pd;
        connect comp3.pd to comp4.pg;
    }
    
    part def SystemDecomposition1 {
        part subsystem1 : b11;
        part subsystem2 : b12;
        connect subsystem1.pe to subsystem2.pf;
    }
    
    part def b21 {
        part comp1 : c1;
        part comp2 : c2;
        part comp3 : c3;
        part comp4 : c4;
        connect comp1.pb to comp2.pc;
        connect comp3.pd to comp4.pg;
    }
    
    part def b22 {
        part comp1 : c1;
        part comp2 : c2;
        part comp3 : c3;
        part comp4 : c4;
        connect comp1.pb to comp3.pd;
        connect comp2.pc to comp4.pg;
    }
    
    part def SystemDecomposition2 {
        part module1 : b21;
        part module2 : b22;
    }
}