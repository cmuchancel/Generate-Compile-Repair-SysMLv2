package CouplingSystemModel {
    private import ScalarValues::*;

    port def CouplingPort;

    part def Ball;

    part def Hitch {
        part ball: Ball;
        port couplingPort: CouplingPort;
    }

    part def Vehicle {
        port hitchPort: CouplingPort;
    }

    part def Trailer {
        port couplingPort: CouplingPort;
    }

    part def CouplingSystem {
        part vehicle: Vehicle;
        part trailer: Trailer;
        part hitch: Hitch;
        connect vehicle.hitchPort to hitch.couplingPort;
        connect trailer.couplingPort to hitch.couplingPort;
    }
}