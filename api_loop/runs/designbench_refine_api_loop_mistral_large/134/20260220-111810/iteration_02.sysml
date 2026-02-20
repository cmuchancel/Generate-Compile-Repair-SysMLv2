package BloodGlucoseMeterSystem {
    private import ScalarValues::*;

    enum def BatteryStatus {
        enum OK;
        enum LOW;
        enum DEPLETED;
        enum CHARGING;
        enum FAULT;
    }

    part def Battery {
        attribute status : BatteryStatus = BatteryStatus::OK;
        attribute voltage : Real;
        attribute chargeLevel : Real;
    }

    port def AlertSignal;

    part def AlertSystem {
        port alertOut : AlertSignal;
    }

    item def Alert {
        attribute message : String;
        attribute severity : String;
    }

    part def BloodGlucoseMeter {
        part battery : Battery;
        part alertSystem : AlertSystem;

        attribute isOperational : Boolean = true;
        attribute lastMeasurementTime : Real;
    }

    part meter : BloodGlucoseMeter;

    connect meter.alertSystem.alertOut to meter.battery;
}