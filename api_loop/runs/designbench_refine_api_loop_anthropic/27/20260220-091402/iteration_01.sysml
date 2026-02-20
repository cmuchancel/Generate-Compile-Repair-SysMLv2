package BatteryChargingControlSystem {
    private import ScalarValues::*;

    enum def ChargingStatus {
        enum CHARGING;
        enum NOT_CHARGING;
        enum TERMINATED;
    }

    item def BatteryLevel {
        attribute level : Real;
    }

    item def ChargingCommand {
        attribute command : ChargingStatus;
    }

    port def BatteryMonitorPort {
        attribute currentLevel : Real;
    }

    port def ChargingControlPort {
        attribute status : ChargingStatus;
    }

    part def BatteryMonitor {
        port monitorOut : BatteryMonitorPort;
        attribute batteryLevel : Real;
        attribute isCharging : Boolean;
        attribute isTerminated : Boolean;
    }

    part def ChargingController {
        port controlIn : BatteryMonitorPort;
        port controlOut : ChargingControlPort;
        attribute targetLevel : Real = 100.0;
        attribute currentStatus : ChargingStatus;
        attribute autoCharge : Boolean;
        attribute safeCharge : Boolean;
    }

    part def BatteryUnit {
        port batteryPort : BatteryMonitorPort;
        attribute capacity : Real;
        attribute currentCharge : Real;
        attribute chargeRate : Real;
        attribute isHealthy : Boolean;
    }

    part def ChargingTerminator {
        port terminatorIn : BatteryMonitorPort;
        port terminatorOut : ChargingControlPort;
        attribute threshold : Real = 100.0;
        attribute terminationTriggered : Boolean;
    }

    part def BatteryChargingSystem {
        part monitor : BatteryMonitor;
        part controller : ChargingController;
        part battery : BatteryUnit;
        part terminator : ChargingTerminator;

        connect monitor.monitorOut to controller.controlIn;
        connect battery.batteryPort to monitor.monitorOut;
        connect terminator.terminatorIn to controller.controlOut;
        connect controller.controlOut to terminator.terminatorIn;
    }
}