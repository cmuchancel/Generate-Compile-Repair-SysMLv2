package VehicleStructure {
    private import ScalarValues::*;
    
    part def Wheel {
    }
    
    part def Axle {
    }
    
    part def SteeringAxle {
        attribute steeringAngle : Real;
    }
    
    part def AxleAssembly {
        part axle : Axle;
        part leftWheel : Wheel;
        part rightWheel : Wheel;
    }
    
    part def SteeringAxleAssembly {
        part axle : SteeringAxle;
        part leftWheel : Wheel;
        part rightWheel : Wheel;
    }
    
    part def Vehicle {
        attribute mass : Real;
        part frontAxleAssembly : SteeringAxleAssembly;
        part rearAxleAssembly : AxleAssembly;
    }
    
    part exampleVehicle1 : Vehicle {
        attribute mass = 1750.0;
    }
    
    part exampleVehicle2 : Vehicle {
        attribute mass = 2000.0;
    }
}