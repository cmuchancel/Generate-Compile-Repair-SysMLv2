package WheelAssembly {
    private import ScalarValues::*;
    
    part def WheelBearingUnit;
    
    part def Tire {
        part bead1 : Bead;
        part bead2 : Bead;
    }
    
    part def Bead;
    
    part def Wheel {
        part rim1 : Rim;
        part rim2 : Rim;
        part hub : Hub;
    }
    
    part def Rim;
    
    part def Hub {
        attribute mountingHoleCount : Integer = 5;
        part mountingHoles[1..mountingHoleCount] : BoltMountingHole;
    }
    
    part def BoltMountingHole;
    
    part def VehicleHub {
        attribute threadedHoleCount : Integer = 5;
        part threadedHoles[1..threadedHoleCount] : ThreadedHole;
    }
    
    part def ThreadedHole;
    
    part def Bolt;
    
    part def BoltConnection {
        part bolt : Bolt;
        port mountingPort : BoltMountingPort;
        port threadedPort : ThreadedPort;
    }
    
    port def BoltMountingPort;
    port def ThreadedPort;
    
    part def WheelAssembly {
        part wheelBearingUnit : WheelBearingUnit;
        part tire : Tire;
        part wheel : Wheel;
        part vehicleHub : VehicleHub;
        part connections[1..5] : BoltConnection;
        
        connect tire.bead1 to wheel.rim1;
        connect tire.bead2 to wheel.rim2;
        
        for i in 1..5 {
            connect wheel.hub.mountingHoles[i] to connections[i].mountingPort;
            connect vehicleHub.threadedHoles[i] to connections[i].threadedPort;
        }
    }
}