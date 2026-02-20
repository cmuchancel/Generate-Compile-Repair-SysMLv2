package BatteryChargingSystem {
    private import ScalarValues::*;

    port def LevelPort {
        attribute level : Real;
    }

    port def ControlPort {
        attribute charging : Boolean;
    }

    part def Battery {
        attribute level : Real;
        attribute charging : Boolean;
        port levelPort : LevelPort;
        port controlPort : ControlPort;
    }

    part def ChargerController {
        attribute targetLevel : Real = 100.0;
        attribute autoStop : Boolean = true;
        port levelPort : LevelPort;
        port controlPort : ControlPort;
    }

    part def ChargingSystem {
        part battery : Battery;
        part controller : ChargerController;
        connect battery.levelPort to controller.levelPort;
        connect controller.controlPort to battery.controlPort;
    }
}