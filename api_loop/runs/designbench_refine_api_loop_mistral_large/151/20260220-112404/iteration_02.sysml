package transportation_system {
    private import ScalarValues::*;

    enum def FuelStatus {
        enum OK
        enum LOW
        enum EMPTY
    }

    enum def EnvironmentalFactor {
        enum RAIN
        enum SNOW
        enum FOG
        enum HIGH_TEMPERATURE
        enum LOW_TEMPERATURE
        enum NORMAL
    }

    part def User {
        attribute role : String;
    }

    part def Driver {
        attribute licenseValid : Boolean = true;
    }

    part def Passenger {
    }

    part def Vehicle {
        attribute maxFuelCapacity : Real;
        attribute currentFuelLevel : Real = maxFuelCapacity;
        attribute fuelStatus : FuelStatus = FuelStatus::OK;
        attribute maxPassengerCapacity : Integer = 4;
        attribute currentPassengerCount : Integer = 0;

        port fuelPort;
        port userEntryPort;
        port environmentalSensorPort;
    }

    part def GasStation {
        port fuelSupplyPort;
    }

    part def Environment {
        attribute currentFactor : EnvironmentalFactor = EnvironmentalFactor::NORMAL;
    }

    part system : Vehicle {
        part driver : Driver;
        part passengers : Passenger[0..4];

        attribute fuelThreshold : Real = 0.1;

        connect driver to userEntryPort;
        connect passengers to userEntryPort;
    }

    part refuelingSystem {
        part vehicle : Vehicle;
        part gasStation : GasStation;

        connect vehicle.fuelPort to gasStation.fuelSupplyPort;
    }

    part environmentalMonitoring {
        part vehicle : Vehicle;
        part environment : Environment;

        connect environment to vehicle.environmentalSensorPort;
    }
}