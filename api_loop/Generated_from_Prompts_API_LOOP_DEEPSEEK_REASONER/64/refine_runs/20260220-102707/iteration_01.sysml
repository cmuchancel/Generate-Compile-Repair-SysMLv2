package VehicleMassCompliance {
    private import ScalarValues::*;
    
    part def Chassis {
        attribute mass : Real;
    }
    
    part def Engine {
        attribute mass : Real;
    }
    
    part def Transmission {
        attribute mass : Real;
    }
    
    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        attribute totalMass : Real;
        attribute massLimit : Real = 2500.0;
        
        item def MassSum {
            attribute sum : Real;
        }
        
        port def massDataPort {
            attribute chassisMass : Real;
            attribute engineMass : Real;
            attribute transmissionMass : Real;
        }
        
        port massData : massDataPort;
        
        connect massData.chassisMass to chassis.mass;
        connect massData.engineMass to engine.mass;
        connect massData.transmissionMass to transmission.mass;
        
        attribute massSumItem : MassSum;
        
        connect chassis.mass to massSumItem.sum;
        connect engine.mass to massSumItem.sum;
        connect transmission.mass to massSumItem.sum;
        connect massSumItem.sum to totalMass;
    }
}