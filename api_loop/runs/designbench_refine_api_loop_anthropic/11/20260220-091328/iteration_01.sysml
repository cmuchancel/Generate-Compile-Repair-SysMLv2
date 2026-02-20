package FuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum Supply;
        enum Return;
    }

    item def Fuel;

    item def FuelWithTemperature {
        attribute temperature : Real;
        attribute flowDirection : FuelFlowDirection;
    }

    port def FuelSupplyPort {
        attribute flowRate : Real;
        attribute pressure : Real;
    }

    port def FuelReturnPort {
        attribute flowRate : Real;
        attribute pressure : Real;
    }

    port def MonitorPort {
        attribute temperature : Real;
        attribute pressure : Real;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port supplyOut : FuelSupplyPort;
        port returnIn : FuelReturnPort;
        port monitorOut : MonitorPort;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
        attribute operatingTemperature : Real;

        port supplyIn : FuelSupplyPort;
        port returnOut : FuelReturnPort;
        port monitorIn : MonitorPort;
    }

    part def FuelLine {
        attribute length : Real;
        attribute diameter : Real;
        attribute maxPressure : Real;

        port lineIn : FuelSupplyPort;
        port lineOut : FuelSupplyPort;
    }

    part def ReturnLine {
        attribute length : Real;
        attribute diameter : Real;
        attribute maxPressure : Real;

        port returnLineIn : FuelReturnPort;
        port returnLineOut : FuelReturnPort;
    }

    part def TemperatureSensor {
        attribute measuredTemperature : Real;
        attribute minOperatingTemp : Real;
        attribute maxOperatingTemp : Real;

        port sensorMonitor : MonitorPort;
    }

    part def FuelPump {
        attribute pumpPressure : Real;
        attribute pumpFlowRate : Real;
        attribute isOperational : Boolean;

        port pumpIn : FuelSupplyPort;
        port pumpOut : FuelSupplyPort;
    }

    part def VehicleFuelSystem {
        attribute vehicleId : String;
        attribute systemOperational : Boolean;

        part tank : FuelTankAssembly {
            attribute capacity = 60.0;
            attribute currentFuelLevel = 45.0;
            attribute fuelTemperature = 25.0;
        }

        part engine : Engine {
            attribute fuelConsumptionRate = 8.5;
            attribute operatingTemperature = 90.0;
        }

        part supplyLine : FuelLine {
            attribute length = 2.5;
            attribute diameter = 0.012;
            attribute maxPressure = 500.0;
        }

        part returnLine : ReturnLine {
            attribute length = 2.5;
            attribute diameter = 0.010;
            attribute maxPressure = 300.0;
        }

        part tempSensor : TemperatureSensor {
            attribute measuredTemperature = 25.0;
            attribute minOperatingTemp = -40.0;
            attribute maxOperatingTemp = 120.0;
        }

        part pump : FuelPump {
            attribute pumpPressure = 400.0;
            attribute pumpFlowRate = 150.0;
            attribute isOperational = true;
        }

        connect tank.supplyOut to pump.pumpIn;
        connect pump.pumpOut to supplyLine.lineIn;
        connect supplyLine.lineOut to engine.supplyIn;
        connect engine.returnOut to returnLine.returnLineIn;
        connect returnLine.returnLineOut to tank.returnIn;
        connect tank.monitorOut to tempSensor.sensorMonitor;
        connect tempSensor.sensorMonitor to engine.monitorIn;
    }
}