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
        port batteryPort : BatteryPort;
    }

    part def Charger {
        port powerOut : PowerPort;
    }

    port def PowerPort {
    }

    port def BatteryPort {
    }

    part def ChargingController {
        attribute currentState : ChargingState = ChargingState::IDLE;
        attribute threshold : Real = 100.0;

        port batteryPort : BatteryPort;
        port chargerPort : PowerPort;
    }

    part system {
        part battery : Battery;
        part charger : Charger;
        part controller : ChargingController;

        connect controller.batteryPort to battery.batteryPort;
        connect controller.chargerPort to charger.powerOut;
    }
}