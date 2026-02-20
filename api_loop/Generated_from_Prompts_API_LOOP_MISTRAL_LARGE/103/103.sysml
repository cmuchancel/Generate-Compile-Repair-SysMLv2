package MultiArchitectureSystem {
    private import ScalarValues::*;

    part def System;

    part def Subsystem;
    part def Module;
    part def Component;

    port def SignalPort;
    port def InterfacePort;

    item def Signal;

    part def System {
        part b11 : Subsystem;
        part b12 : Subsystem;

        part b21 : Module;
        part b22 : Module;
    }

    part def Subsystem {
        part c1 : Component;
        part c2 : Component;
        part c3 : Component;
        part c4 : Component;

        port pe : InterfacePort;
        port pf : InterfacePort;
    }

    part def Module {
        part c1 : Component;
        part c2 : Component;
        part c3 : Component;
        part c4 : Component;
    }

    part def Component {
        port pb : SignalPort;
        port pd : SignalPort;
        port internal : SignalPort;
    }

    part system : System {
        part b11 : Subsystem {
            part c1 : Component {
                port pb : SignalPort;
            }
            part c2 : Component {
                port internal : SignalPort;
            }
            port pe : InterfacePort;
        }

        part b12 : Subsystem {
            part c3 : Component {
                port pd : SignalPort;
            }
            part c4 : Component;
            port pf : InterfacePort;
        }

        part b21 : Module {
            part c1 :> system.b11.c1;
            part c2 :> system.b11.c2;
            part c3 :> system.b12.c3;
            part c4 :> system.b12.c4;
        }

        part b22 : Module {
            part c1 :> system.b11.c1;
            part c2 :> system.b11.c2;
            part c3 :> system.b12.c3;
            part c4 :> system.b12.c4;
        }

        connect system.b11.c1.pb to system.b11.c2.internal;
        connect system.b11.pe to system.b11.c1.pb;
        connect system.b12.pf to system.b12.c3.pd;
        connect system.b11.pe to system.b12.pf;
    }
}