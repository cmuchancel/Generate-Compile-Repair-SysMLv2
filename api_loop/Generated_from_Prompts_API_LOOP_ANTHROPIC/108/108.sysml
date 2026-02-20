package VehicleTrailerSystem {
    private import ScalarValues::*;

    enum def ConnectionStatus {
        enum Connected;
        enum Disconnected;
    }

    part def TowBall {
        attribute status : ConnectionStatus;
    }

    part def Coupler {
        attribute status : ConnectionStatus;
    }

    part def Chassis {
        part towBall : TowBall;
    }

    part def TrailerFrame {
        part coupler : Coupler;
    }

    part def Vehicle {
        part chassis : Chassis;
    }

    part def Trailer {
        part trailerFrame : TrailerFrame;
    }

    part def HitchSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        attribute hitchStatus : ConnectionStatus;
    }

    part hitchSystem : HitchSystem {
        part vehicle : Vehicle {
            part chassis : Chassis {
                part towBall : TowBall {
                    attribute status : ConnectionStatus = ConnectionStatus::Disconnected;
                }
            }
        }
        part trailer : Trailer {
            part trailerFrame : TrailerFrame {
                part coupler : Coupler {
                    attribute status : ConnectionStatus = ConnectionStatus::Disconnected;
                }
            }
        }
        attribute hitchStatus : ConnectionStatus = ConnectionStatus::Disconnected;

        connect vehicle.chassis.towBall to trailer.trailerFrame.coupler;
    }
}