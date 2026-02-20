package BatteryChargingControlSystem {
    private import ScalarValues::*;

    enum def ChargingState {
        enum CHARGING,
        enum NOT_CHARGING
    }

    part def Battery {
        attribute level : Real = 0.0;
        port batteryPort : BatteryPort;
    }

    part def Charger {
        attribute state : ChargingState = ChargingState::NOT_CHARGING;
        port chargerPort : ChargerPort;
    }

    part def Controller {
        attribute batteryLevel : Real = 0.0;
        attribute desiredState : ChargingState = ChargingState::NOT_CHARGING;
        port batteryPort : BatteryPort;
        port chargerPort : ChargerPort;
    }

    port def BatteryPort {
        attribute level : Real;
    }

    port def ChargerPort {
        attribute state : ChargingState;
    }

    part battery1 : Battery;
    part charger1 : Charger;
    part controller1 : Controller;

    connect controller1.batteryPort to battery1.batteryPort;
    connect controller1.chargerPort to charger1.chargerPort;
}