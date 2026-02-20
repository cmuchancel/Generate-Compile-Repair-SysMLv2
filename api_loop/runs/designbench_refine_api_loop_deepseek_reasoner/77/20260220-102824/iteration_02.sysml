package TransportationSystem {
    private import ScalarValues::*;

    part def TransportationSystem {
        part vehicle : Vehicle;
        part environment : Environment;
        part infrastructure : Infrastructure;
    }

    part def Vehicle {
        attribute maxPassengers : Integer = 4;
        part driver : User[0..1];
        part passengers : User[0..4];
        port entryExit : EntryExit;
        port environmentInterface : EnvironmentInterface;
        port infrastructureInterface : InfrastructureInterface;
    }

    part def User {
        port access : EntryExit;
    }

    part def Environment {
        port sensorData : EnvironmentInterface;
    }

    part def Infrastructure {
        port services : InfrastructureInterface;
    }

    port def EntryExit {}

    port def EnvironmentInterface {}

    port def InfrastructureInterface {}

    part mySystem : TransportationSystem {
        connect vehicle.driver.access to vehicle.entryExit;
        connect vehicle.passengers.access to vehicle.entryExit;
        connect vehicle.environmentInterface to environment.sensorData;
        connect vehicle.infrastructureInterface to infrastructure.services;
    }
}