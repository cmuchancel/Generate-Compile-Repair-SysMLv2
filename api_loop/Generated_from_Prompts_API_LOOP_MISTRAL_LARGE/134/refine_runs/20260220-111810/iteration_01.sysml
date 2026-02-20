package BloodGlucoseMeterSystem {
    private import ScalarValues::*;

    enum def BatteryStatus {
        OK
        LOW
        DEPLETED
        CHARGING
        FAULT
    }

    part def Battery {
        attribute status : BatteryStatus = BatteryStatus::OK;
        attribute voltage : Real;
        attribute chargeLevel : Real;
    }

    part def AlertSystem {
        port alertOut : AlertSignal;
    }

    port def AlertSignal;

    part def BloodGlucoseMeter {
        part battery : Battery;
        part alertSystem : AlertSystem;

        attribute isOperational : Boolean = true;
        attribute lastMeasurementTime : Real;
    }

    item def Alert {
        attribute message : String;
        attribute severity : String;
    }

    part meter : BloodGlucoseMeter;

    connect meter.battery to meter.alertSystem.alertOut;
}