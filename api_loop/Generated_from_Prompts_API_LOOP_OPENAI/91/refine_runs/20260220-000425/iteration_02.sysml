package BatteryRiskSystem {
    private import ScalarValues::*;

    enum def SeverityLevel {
        LOW;
        MEDIUM;
        HIGH;
        CRITICAL;
    }

    enum def BatteryState {
        NORMAL;
        LOW_BATTERY;
        SHUTDOWN;
    }

    enum def RiskType {
        NONE;
        BATTERY_AGING_TO_LOW_LEVEL;
        LOW_BATTERY;
        DEVICE_SHUTDOWN;
    }

    item def BatteryStatus {
        attribute level : Real;
        attribute minThreshold : Real;
        attribute agingProbability : Real;
    }

    part def Battery {
        attribute level : Real;
        attribute minThreshold : Real;
        attribute agingProbability : Real;
        attribute batteryState : BatteryState;
    }

    part def RiskAssessment {
        attribute riskIdentified : Boolean;
        attribute riskType : RiskType;
        attribute severity : SeverityLevel;
        attribute lowBatteryState : Boolean;
        attribute prolongedLowBattery : Boolean;
        attribute deviceShutdown : Boolean;
    }

    part def EquipmentSystem {
        part battery : Battery;
        part assessment : RiskAssessment;
        attribute monitorEnabled : Boolean;
        attribute criticalFunctionInterrupted : Boolean;
    }

    part equipment : EquipmentSystem;
}