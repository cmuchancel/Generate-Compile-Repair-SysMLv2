package BatteryChargingSystem {
    private import ScalarValues::*;

    enum def ChargingStatus {
        enum CHARGING;
        enum COMPLETE;
        enum MONITORING;
    }

    item def BatteryLevel {
        attribute percentage : Real;
    }

    item def ChargingCommand {
        attribute activate : Boolean;
    }

    port def BatteryMonitorPort {
        attribute currentLevel : Real;
    }

    port def ChargingControlPort {
        attribute chargingActive : Boolean;
    }

    part def BatteryMonitor {
        attribute measuredLevel : Real = 0.0;
        attribute threshold : Real = 100.0;
        attribute belowThreshold : Boolean = true;
        port monitorOut : BatteryMonitorPort;
    }

    part def ChargingController {
        attribute status : ChargingStatus = ChargingStatus::MONITORING;
        attribute chargingEnabled : Boolean = true;
        port controlOut : ChargingControlPort;
        port monitorIn : BatteryMonitorPort;
    }

    part def BatteryReplenisher {
        attribute replenishRate : Real = 1.0;
        attribute active : Boolean = false;
        port replenishControl : ChargingControlPort;
    }

    part def Battery {
        attribute level : Real = 0.0;
        attribute capacity : Real = 100.0;
        attribute fullyCharged : Boolean = false;
        port batteryMonitorPort : BatteryMonitorPort;
        port batteryChargePort : ChargingControlPort;
    }

    part def ChargingSystem {
        attribute systemActive : Boolean = true;
        attribute overchargeProtection : Boolean = true;
        attribute automatedOperation : Boolean = true;

        part monitor : BatteryMonitor;
        part controller : ChargingController;
        part replenisher : BatteryReplenisher;
        part battery : Battery;

        connect monitor.monitorOut to controller.monitorIn;
        connect controller.controlOut to replenisher.replenishControl;
        connect battery.batteryMonitorPort to monitor.monitorOut;
        connect battery.batteryChargePort to replenisher.replenishControl;
    }
}