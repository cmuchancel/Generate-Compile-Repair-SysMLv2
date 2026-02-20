package TransportationSystem {
        private import ScalarValues::*;

        part def TransportationSystem {
            part vehicle : Vehicle;
            part environment : Environment;
            part infrastructure : Infrastructure;
            part users : User[0..5];
        }

        part def Vehicle {
            attribute maxPassengers : Integer = 4;
            part driver : Driver[0..1];
            part passengers : Passenger[0..4];
            port entryExit : EntryExit;
            port environmentInterface : EnvironmentInterface;
            port infrastructureInterface : InfrastructureInterface;
        }

        part def User {}

        part def Driver {
            port access : EntryExit;
        }

        part def Passenger {
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
    }

    part TransportationSystem {
        part vehicle : Vehicle;
        part environment : Environment;
        part infrastructure : Infrastructure;
        part users : User[5];

        connect users[0].access to vehicle.entryExit;
        connect users[1].access to vehicle.entryExit;
        connect users[2].access to vehicle.entryExit;
        connect users[3].access to vehicle.entryExit;
        connect users[4].access to vehicle.entryExit;
        connect vehicle.environmentInterface to environment.sensorData;
        connect vehicle.infrastructureInterface to infrastructure.services;
    }
}