package WheelAssembly {
    private import ScalarValues::*;

    part def WheelBearingUnit;

    part def Tire {
        part bead1 : Bead;
        part bead2 : Bead;
    }

    part def Bead;

    part def Wheel {
        part rim1 : Rim;
        part rim2 : Rim;
        part hub : Hub;
    }

    part def Rim;

    part def Hub {
        attribute mountingHoleCount : Integer = 5;
        part mountingHole1 : BoltMountingHole;
        part mountingHole2 : BoltMountingHole;
        part mountingHole3 : BoltMountingHole;
        part mountingHole4 : BoltMountingHole;
        part mountingHole5 : BoltMountingHole;
    }

    part def BoltMountingHole;

    part def VehicleHub {
        attribute threadedHoleCount : Integer = 5;
        part threadedHole1 : ThreadedHole;
        part threadedHole2 : ThreadedHole;
        part threadedHole3 : ThreadedHole;
        part threadedHole4 : ThreadedHole;
        part threadedHole5 : ThreadedHole;
    }

    part def ThreadedHole;

    part def Bolt;

    part def BoltConnection {
        part bolt : Bolt;
        port mountingPort : BoltMountingPort;
        port threadedPort : ThreadedPort;
    }

    port def BoltMountingPort;
    port def ThreadedPort;

    part def WheelAssembly {
        part wheelBearingUnit : WheelBearingUnit;
        part tire : Tire;
        part wheel : Wheel;
        part vehicleHub : VehicleHub;
        part connection1 : BoltConnection;
        part connection2 : BoltConnection;
        part connection3 : BoltConnection;
        part connection4 : BoltConnection;
        part connection5 : BoltConnection;

        connect tire.bead1 to wheel.rim1;
        connect tire.bead2 to wheel.rim2;
        connect wheel.hub.mountingHole1 to connection1.mountingPort;
        connect vehicleHub.threadedHole1 to connection1.threadedPort;
        connect wheel.hub.mountingHole2 to connection2.mountingPort;
        connect vehicleHub.threadedHole2 to connection2.threadedPort;
        connect wheel.hub.mountingHole3 to connection3.mountingPort;
        connect vehicleHub.threadedHole3 to connection3.threadedPort;
        connect wheel.hub.mountingHole4 to connection4.mountingPort;
        connect vehicleHub.threadedHole4 to connection4.threadedPort;
        connect wheel.hub.mountingHole5 to connection5.mountingPort;
        connect vehicleHub.threadedHole5 to connection5.threadedPort;
    }
}