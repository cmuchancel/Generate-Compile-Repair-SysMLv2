package UnitSystemLibrary
    private import ScalarValues::*;
    
    enum def UnitSystemKind { 
        ISQ;
        USCustomary;
    }
    
    enum def QuantityKind {
        Length;
        Mass;
        Time;
        Temperature;
        Area;
        Volume;
        Force;
        Energy;
        Power;
        Pressure;
    }
    
    item def Unit {
        attribute name : String;
        attribute symbol : String;
        attribute system : UnitSystemKind;
        attribute quantity : QuantityKind;
        attribute toBaseFactor : Real;
    }
    
    item def UnitConversion {
        attribute fromUnit : String;
        attribute toUnit : String;
        attribute factor : Real;
        attribute offset : Real;
    }
    
    part def UnitCatalog {
        attribute name : String;
        part units : Unit[0..*];
        part conversions : UnitConversion[0..*];
    }
    
    part def UnitLibrary {
        part catalog : UnitCatalog;
        attribute version : String;
    }
    
    part def UnitSystemService {
        part library : UnitLibrary;
    }
    
    part library : UnitLibrary;
    part service : UnitSystemService;
    connect library to service.library;
end package