package BatteryChargingControlSystem {
    private import ScalarValues::*;

    enum def ChargingState {
        enum IDLE;
        enum CHARGING;
        enum FULL;
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

    part def ChargeController {
        attribute chargingState : ChargingState = ChargingState::IDLE;
        attribute threshold : Real = 100.0;

        part battery : Battery;
        part charger : Charger;

        port monitor : MonitorPort;
        port control : ControlPort;
    }

    port def MonitorPort {
    }

    port def ControlPort {
    }

    part system : ChargeController;

    connect system.battery to system.monitor;
    connect system.charger.powerOut to system.control;
}