package BatteryChargingSystem {
    private import ScalarValues::*;

    part def Battery {
        attribute level : Real = 0.0;
    }

    part def Charger {
        port charging : ChargerPort;
        attribute targetLevel : Real = 100.0;
    }

    port def ChargerPort {
        attribute connectedLevel : Real;
        attribute command : ChargerCommand;
    }

    enum def ChargerCommand {
        START;
        STOP;
    }

    part battery : Battery;
    part charger : Charger;

    connect battery to charger.charging;
}