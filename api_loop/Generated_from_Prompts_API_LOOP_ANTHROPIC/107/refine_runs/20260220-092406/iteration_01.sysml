package VehicleTrailerHitchSystem {
    private import ScalarValues::*;

    enum def ConnectionState {
        enum CONNECTED;
        enum DISCONNECTED;
    }

    item def PowerSignal;

    port def VehiclePowerPort {
        out item power : PowerSignal;
    }

    port def TrailerPowerPort {
        in item power : PowerSignal;
    }

    part def VehicleFrame {
        attribute frameId : String;
    }

    part def TrailerFrame {
        attribute frameId : String;
    }

    part def HitchBall {
        attribute diameter : Real;
        attribute material : String;
    }

    part def TrailerCoupler {
        attribute diameter : Real;
        attribute material : String;
    }

    part def HitchMechanism {
        attribute state : ConnectionState;
        port vehicleSide : VehiclePowerPort;
        port trailerSide : TrailerPowerPort;
        part hitchBall : HitchBall {
            attribute diameter : Real = 1.875;
            attribute material : String = "Steel";
        }
        part trailerCoupler : TrailerCoupler {
            attribute diameter : Real = 1.875;
            attribute material : String = "Steel";
        }
    }

    part def Vehicle {
        attribute vehicleId : String;
        part frame : VehicleFrame {
            attribute frameId : String = "VF-001";
        }
        part hitch : HitchMechanism;
        port powerOut : VehiclePowerPort;
    }

    part def Trailer {
        attribute trailerId : String;
        part frame : TrailerFrame {
            attribute frameId : String = "TF-001";
        }
        port powerIn : TrailerPowerPort;
    }

    part def HitchSystem {
        attribute systemName : String = "VehicleTrailerHitchSystem";
        attribute connectionStatus : ConnectionState;

        part vehicle : Vehicle {
            attribute vehicleId : String = "V-001";
        }

        part trailer : Trailer {
            attribute trailerId : String = "T-001";
        }

        connect vehicle.powerOut to trailer.powerIn;
    }
}