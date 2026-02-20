package VehicleTrailerCouplingSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum Connected;
        enum Disconnected;
    }

    part def CouplingBall {
        attribute diameter : Real;
        attribute material : String;
    }

    part def TrailerConnector {
        attribute socketSize : Real;
        attribute material : String;
    }

    part def TowHitch {
        part ball : CouplingBall;
        part connector : TrailerConnector;
        attribute status : ConnectionStatus;
    }

    part def Vehicle {
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        port hitchPort : HitchPort;
    }

    part def Trailer {
        attribute make : String;
        attribute model : String;
        attribute loadCapacity : Real;
        port trailerPort : TrailerPort;
    }

    port def HitchPort {
        attribute compatibility : String;
    }

    port def TrailerPort {
        attribute compatibility : String;
    }

    part def CouplingSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        part hitch : TowHitch;

        attribute systemName : String = "VehicleTrailerCouplingSystem";

        connect vehicle.hitchPort to trailer.trailerPort;
    }

    part couplingSystemInstance : CouplingSystem {
        part vehicle : Vehicle {
            attribute make : String = "Toyota";
            attribute model : String = "Tundra";
            attribute year : Integer = 2023;
        }

        part trailer : Trailer {
            attribute make : String = "Airstream";
            attribute model : String = "Classic";
            attribute loadCapacity : Real = 5000.0;
        }

        part hitch : TowHitch {
            part ball : CouplingBall {
                attribute diameter : Real = 2.0;
                attribute material : String = "Steel";
            }
            part connector : TrailerConnector {
                attribute socketSize : Real = 2.0;
                attribute material : String = "Steel";
            }
            attribute status : ConnectionStatus = ConnectionStatus::Connected;
        }

        connect vehicle.hitchPort to trailer.trailerPort;
    }
}