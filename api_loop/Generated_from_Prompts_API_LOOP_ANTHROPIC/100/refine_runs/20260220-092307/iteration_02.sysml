package VehicleConfiguration {
    private import ScalarValues::*;

    part def Tire {
        attribute mass : Real;
    }

    part def Axle {
        attribute mass : Real;
        part tires : Tire[2] ordered;
    }

    part def FrontAxle :> Axle {
        attribute steeringAngle : Real;
    }

    part def RearAxle :> Axle {
    }

    part def Vehicle {
        attribute mass : Real;
        part frontAxle : Axle[1];
        part rearAxle : Axle[1];
    }

    part vehicle1 : Vehicle {
        attribute :>> mass = 1750;

        part :>> frontAxle : FrontAxle {
            attribute :>> mass = 100;
            part :>> tires : Tire[2] ordered {
                attribute :>> mass = 10;
            }
        }

        part :>> rearAxle : RearAxle {
            attribute :>> mass = 120;
            part :>> tires : Tire[2] ordered {
                attribute :>> mass = 10;
            }
        }
    }

    part vehicle1_c1 : Vehicle {
        attribute :>> mass = 2000;

        part frontAxle_c1 : FrontAxle {
            attribute :>> mass = 110;
            attribute :>> steeringAngle = 30.0;
            part tires_front : Tire[2] ordered {
                attribute :>> mass = 12;
            }
        }

        part rearAxle_c1 : RearAxle {
            attribute :>> mass = 130;
            part tires_rear : Tire[2] ordered {
                attribute :>> mass = 12;
            }
        }
    }
}