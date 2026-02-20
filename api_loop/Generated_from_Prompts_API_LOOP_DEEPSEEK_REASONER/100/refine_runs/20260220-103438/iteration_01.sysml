package VehicleVariants {
    private import ScalarValues::*;
    
    part def Tire {
        attribute mass : kg;
    }
    
    part def Axle {
        attribute mass : kg;
        part leftTire : Tire[1];
        part rightTire : Tire[1];
    }
    
    part def SteeringAxle :> Axle {
        attribute steeringAngle : degrees;
    }
    
    part def Vehicle {
        attribute mass : kg = 1750;
        part frontAxle : Axle[1];
        part rearAxle : Axle[1];
    }
    
    part def vehicle1_c1 :> Vehicle {
        attribute mass : kg = 2000;
        part frontAxle : SteeringAxle[1];
    }
}