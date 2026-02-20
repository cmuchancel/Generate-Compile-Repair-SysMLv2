package VehicleMassManagement {
    private import ScalarValues::*;
    
    part def Component {
        attribute serialNumber : String;
        attribute mass : Real;
        part subcomponents : Component[0..*];
    }
    
    part def Engine :> Component;
    part def Transmission :> Component;
    
    part def Vehicle {
        attribute vin : String;
        attribute ownMass : Real;
        attribute massThreshold : Real = 0.0;
        part engine : Engine;
        part transmission : Transmission;
        part components : Component[0..*];
        
        attribute totalMass : Real;
        constraint {
            totalMass = ownMass + 
                engine.mass + 
                transmission.mass + 
                sum(components->select(c|c.mass >= massThreshold).mass) +
                sum(components->select(c|c.mass >= massThreshold).subcomponents->select(sc|sc.mass >= massThreshold).mass);
        }
    }
}