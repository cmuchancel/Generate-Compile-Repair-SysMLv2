package AlloyModel {
    private import ScalarValues::*;
    
    part def Substance {
        attribute name : String;
    }
    
    part def Metal {
        attribute atomicMass : Real;
    }
    
    part def Material {
        attribute name : String;
    }
    
    item def MassFraction {
        attribute fractionValue : Real;
        part component : Metal;
    }
    
    part def Alloy {
        attribute name : String;
        attribute tensileStrength : Real;
        part composition : MassFraction[2..*];
    }
    
    part def MaterialDatabase {
        part iron : Metal;
        attribute iron.atomicMass = 55.845;
        
        part carbon : Metal;
        attribute carbon.atomicMass = 12.011;
        
        part manganese : Metal;
        attribute manganese.atomicMass = 54.938;
        
        part steel980 : Alloy;
        attribute steel980.name = "Steel 980";
        attribute steel980.tensileStrength = 980.0;
        
        part comp1 : MassFraction;
        attribute comp1.fractionValue = 0.98;
        part comp2 : MassFraction;
        attribute comp2.fractionValue = 0.02;
        
        connect steel980.composition to comp1;
        connect steel980.composition to comp2;
        connect comp1.component to iron;
        connect comp2.component to carbon;
    }
}