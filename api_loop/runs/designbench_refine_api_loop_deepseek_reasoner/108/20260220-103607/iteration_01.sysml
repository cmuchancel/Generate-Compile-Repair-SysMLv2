package VehicleTrailerSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum CONNECTED;
        enum DISCONNECTED;
    }

    part def TowBall {
        attribute isAvailable : Boolean = true;
    }

    part def Coupler {
        attribute isEngaged : Boolean = false;
    }

    part def Chassis {
        part towBall : TowBall;
    }

    part def TrailerFrame {
        part coupler : Coupler;
    }

    part def Vehicle {
        part chassis : Chassis;
        attribute vehicleID : String;
    }

    part def Trailer {
        part frame : TrailerFrame;
        attribute trailerID : String;
    }

    part def ConnectionManager {
        attribute currentStatus : ConnectionStatus = DISCONNECTED;
        part vehicle : Vehicle;
        part trailer : Trailer;
    }
}