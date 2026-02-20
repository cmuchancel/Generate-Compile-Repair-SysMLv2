package BloodGlucoseMeterSystem {
    private import ScalarValues::*;

    enum def BatteryStatus {
        enum NORMAL;
        enum LOW;
        enum DEPLETED;
    }

    enum def AlarmState {
        enum OFF;
        enum ON;
    }

    part def Battery {
        attribute level : Real = 100.0;
        attribute status : BatteryStatus = BatteryStatus::NORMAL;
        port def powerOutput;
        port powerOutput : powerOutput;
    }

    part def Alarm {
        attribute state : AlarmState = AlarmState::OFF;
        port def triggerInput;
        port triggerInput : triggerInput;
    }

    part def MeasurementModule {
        attribute operational : Boolean = true;
        port def powerInput;
        port def alarmOutput;
        port powerInput : powerInput;
        port alarmOutput : alarmOutput;
    }

    part def BloodGlucoseMeter {
        part battery : Battery;
        part alarm : Alarm;
        part measurementModule : MeasurementModule;

        connect battery.powerOutput to measurementModule.powerInput;
        connect measurementModule.alarmOutput to alarm.triggerInput;
    }
}