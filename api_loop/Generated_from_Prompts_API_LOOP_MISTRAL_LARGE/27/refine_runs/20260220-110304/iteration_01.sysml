package BatteryChargingSystem {
    private import ScalarValues::*;

    enum def ChargingState {
        enum IDLE
        enum CHARGING
        enum FULL
    }

    part def Battery {
        attribute capacity : Real;
        attribute currentLevel : Real = 0.0;
        attribute isCharging : Boolean = false;
    }

    part def Charger {
        port powerOut : PowerPort;
    }

    port def PowerPort {
    }

    part def ChargingController {
        attribute state : ChargingState = ChargingState::IDLE;
        attribute threshold : Real = 100.0;

        port batteryPort : BatteryPort;
        port chargerPort : PowerPort;
    }

    port def BatteryPort {
    }

    part system :> ChargingController {
        part battery :> Battery;
        part charger :> Charger;

        connect batteryPort to battery;
        connect chargerPort to charger.powerOut;
    }
}