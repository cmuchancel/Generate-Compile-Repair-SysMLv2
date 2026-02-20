package CouplingSystem {
    private import ScalarValues::*;
    
    port def MechanicalInterface;
    
    part def Hitch {
        port ball : MechanicalInterface;
        attribute maxLoad : Real;
    }
    
    part def Vehicle {
        port receiver : MechanicalInterface;
        attribute towingCapacity : Real;
    }
    
    part def Trailer {
        port coupler : MechanicalInterface;
        attribute weight : Real;
    }
    
    part def CouplingSystem {
        part vehicle : Vehicle;
        part trailer : Trailer;
        part hitch : Hitch;
        
        connect vehicle.receiver to hitch.ball;
        connect trailer.coupler to hitch.ball;
    }
}