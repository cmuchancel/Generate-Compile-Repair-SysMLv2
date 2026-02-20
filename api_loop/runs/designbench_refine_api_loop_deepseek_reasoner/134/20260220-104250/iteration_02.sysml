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
        port def powerOutputPort;
        port powerOutput : powerOutputPort;
    }

    part def Alarm {
        attribute alarmState : AlarmState = AlarmState::OFF;
        port def triggerInputPort;
        port triggerInput : triggerInputPort;
    }

    part def MeasurementModule {
        attribute operational : Boolean = true;
        port def powerInputPort;
        port def alarmOutputPort;
        port powerInput : powerInputPort;
        port alarmOutput : alarmOutputPort;
    }

    part def BloodGlucoseMeter {
        part battery : Battery;
        part alarm : Alarm;
        part measurementModule : MeasurementModule;

        connect battery.powerOutput to measurementModule.powerInput;
        connect measurementModule.alarmOutput to alarm.triggerInput;
    }
}