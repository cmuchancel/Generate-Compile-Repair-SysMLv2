package transportation_system {
    private import ScalarValues::*;

    enum def PassengerCapacity {
        enum ONE;
        enum TWO;
        enum THREE;
        enum FOUR;
    }

    enum def EnvironmentalFactor {
        enum WEATHER;
        enum ROAD_CONDITION;
        enum TRAFFIC;
        enum AIR_QUALITY;
    }

    part def Vehicle {
        attribute capacity : PassengerCapacity = PassengerCapacity::FOUR;
        attribute driverPresent : Boolean;
        attribute passengers : Integer;
        attribute isOperational : Boolean = true;

        port def EntryExitPort;
        port entryExit : EntryExitPort;

        port def EnvironmentalSensorPort;
        port environmentalSensor : EnvironmentalSensorPort;

        port def InfrastructureInterfacePort;
        port infrastructureInterface : InfrastructureInterfacePort;
    }

    part def User {
        attribute isDriver : Boolean;
    }

    part def Environment {
        attribute factors : EnvironmentalFactor[*];
    }

    part def Infrastructure {
        attribute hasGasStation : Boolean = true;
    }

    part system : Vehicle {
        part driver : User;
        part passengers : User[0..4];

        part environment : Environment;
        part infrastructure : Infrastructure;
    }

    connect system.entryExit to system.driver;
    connect system.entryExit to system.passengers;
    connect system.environmentalSensor to system.environment;
    connect system.infrastructureInterface to system.infrastructure;
}