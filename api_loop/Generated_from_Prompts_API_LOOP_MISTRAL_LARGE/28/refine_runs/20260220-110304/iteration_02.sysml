package BatteryChargingSystem {
    private import ScalarValues::*;

    enum def ChargingState {
        enum MONITORING;
        enum CHARGING;
        enum TERMINATED;
    }

    part def Battery {
        attribute level : Real;
        attribute chargingState : ChargingState = ChargingState::MONITORING;
    }

    part def Charger {
        port powerOut : PowerPort;
    }

    port def PowerPort;

    part def MonitoringSystem {
        attribute currentLevel : Real;
        attribute targetLevel : Real = 100.0;
        attribute threshold : Real = 100.0;

        part battery : Battery;
        part charger : Charger;

        connect battery to charger.powerOut;
    }

    part system : MonitoringSystem;

    item def ChargingCycle {
        attribute isActive : Boolean = true;
    }
}