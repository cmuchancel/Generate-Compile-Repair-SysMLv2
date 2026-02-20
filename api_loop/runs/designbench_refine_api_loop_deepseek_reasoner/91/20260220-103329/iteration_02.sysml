package BatteryRiskAnalysis {
    private import ScalarValues::*;

    enum def DeviceState {
        enum Normal;
        enum LowBattery;
        enum Shutdown;
    }

    enum def Severity {
        enum Low;
        enum Medium;
        enum High;
    }

    part def Battery {
        attribute level : Real;
        attribute agingFactor : Real;
        attribute failureProbability : Real;
    }

    item def Risk {
        attribute description : String;
        attribute severity : Severity;
    }

    item def FailureConsequence {
        attribute description : String;
        attribute severity : Severity;
    }

    part def Equipment {
        attribute state : DeviceState;
        attribute minThreshold : Real = 15.0;
        attribute monitoringActive : Boolean = true;
        part battery : Battery;
        attribute identifiedRisk : Risk;
        attribute failureConsequence : FailureConsequence;
    }
}