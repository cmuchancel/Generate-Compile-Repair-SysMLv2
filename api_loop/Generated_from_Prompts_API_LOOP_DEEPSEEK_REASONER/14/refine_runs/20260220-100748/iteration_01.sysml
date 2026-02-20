package FuelSystem {
    private import ScalarValues::*;
    
    item def Fuel;
    item def TemperatureInfo;
    
    port def FuelOutlet {
        attribute fuel : Fuel;
    }
    
    port def FuelReturn {
        attribute fuel : Fuel;
    }
    
    port def TemperaturePort {
        attribute info : TemperatureInfo;
    }
    
    part def FuelTank {
        port outlet : FuelOutlet;
        port return : FuelReturn;
        port temperature : TemperaturePort;
    }
    
    part def FuelPump {
        port inlet : FuelReturn;
        port outlet : FuelOutlet;
        port return : FuelReturn;
    }
    
    part def FuelTankAssembly {
        port mainOutlet : FuelOutlet;
        port mainReturn : FuelReturn;
        port tempSensor : TemperaturePort;
        
        part tank : FuelTank;
        part pump : FuelPump;
        
        connect tank.outlet to pump.inlet;
        connect pump.outlet to mainOutlet;
        connect mainReturn to pump.return;
        connect pump.return to tank.return;
        connect tank.temperature to tempSensor;
    }
}