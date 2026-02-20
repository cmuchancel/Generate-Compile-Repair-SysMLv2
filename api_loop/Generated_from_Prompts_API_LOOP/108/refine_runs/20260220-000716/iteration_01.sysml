package HitchingSystemModel {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum CONNECTED;
        enum DISCONNECTED;
    }

    port def HitchPort;

    part def TowBall {
        port hitchPort: HitchPort;
    }

    part def Coupler {
        port hitchPort: HitchPort;
    }

    part def Chassis {
        part towBall: TowBall;
    }

    part def TrailerFrame {
        part coupler: Coupler;
    }

    part def Vehicle {
        part chassis: Chassis;
    }

    part def Trailer {
        part frame: TrailerFrame;
    }

    part def HitchingSystem {
        part vehicle: Vehicle;
        part trailer: Trailer;
        attribute status: ConnectionStatus = DISCONNECTED;
        connect vehicle.chassis.towBall.hitchPort to trailer.frame.coupler.hitchPort;
    }
}