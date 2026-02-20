package BatteryRiskAnalysis {
    private import ScalarValues::*;

    enum def SeverityLevel {
        enum LOW;
        enum MEDIUM;
        enum HIGH;
        enum CRITICAL;
    }

    enum def BatteryState {
        enum NORMAL;
        enum LOW_BATTERY;
        enum SHUTDOWN;
    }

    enum def RiskState {
        enum NO_RISK;
        enum AGING_DETECTED;
        enum FAILURE_PROBABLE;
    }

    part def Battery {
        attribute batteryLevel : Real;
        attribute minimumThreshold : Real;
        attribute agingFactor : Real;
        attribute currentState : BatteryState;
        attribute failureProbability : Real;
    }

    part def RiskMonitor {
        attribute monitoringActive : Boolean;
        attribute detectedRisk : RiskState;
        attribute riskDescription : String;
        attribute severityLevel : SeverityLevel;
    }

    part def FailureScenario {
        attribute scenarioName : String;
        attribute cause : String;
        attribute consequence : String;
        attribute severityLevel : SeverityLevel;
        attribute probabilityThreshold : Real;
    }

    part def DeviceShutdownEvent {
        attribute isShutdown : Boolean;
        attribute shutdownCause : String;
        attribute durationBelowThreshold : Real;
        attribute shutdownTriggerDuration : Real;
    }

    part def AlertSystem {
        attribute alertActive : Boolean;
        attribute alertMessage : String;
        attribute alertSeverity : SeverityLevel;
        attribute notifyUser : Boolean;
        attribute notifyMaintenance : Boolean;
    }

    port def BatteryDataPort {
        attribute level : Real;
        attribute state : BatteryState;
    }

    port def RiskDataPort {
        attribute risk : RiskState;
        attribute severity : SeverityLevel;
    }

    part def Equipment {
        attribute equipmentId : String;
        attribute operationalStatus : Boolean;

        part battery : Battery {
            attribute batteryLevel : Real = 100.0;
            attribute minimumThreshold : Real = 20.0;
            attribute agingFactor : Real = 0.0;
            attribute currentState : BatteryState = BatteryState::NORMAL;
            attribute failureProbability : Real = 0.0;
        }

        part riskMonitor : RiskMonitor {
            attribute monitoringActive : Boolean = true;
            attribute detectedRisk : RiskState = RiskState::NO_RISK;
            attribute riskDescription : String = "No risk detected";
            attribute severityLevel : SeverityLevel = SeverityLevel::LOW;
        }

        part batteryAgingScenario : FailureScenario {
            attribute scenarioName : String = "Battery Aging Failure";
            attribute cause : String = "Battery aging increases failure probability";
            attribute consequence : String = "Low battery level";
            attribute severityLevel : SeverityLevel = SeverityLevel::HIGH;
            attribute probabilityThreshold : Real = 0.7;
        }

        part lowBatteryScenario : FailureScenario {
            attribute scenarioName : String = "Low Battery State";
            attribute cause : String = "Battery level below minimum threshold";
            attribute consequence : String = "Device shutdown";
            attribute severityLevel : SeverityLevel = SeverityLevel::CRITICAL;
            attribute probabilityThreshold : Real = 0.9;
        }

        part shutdownEvent : DeviceShutdownEvent {
            attribute isShutdown : Boolean = false;
            attribute shutdownCause : String = "Prolonged low battery state";
            attribute durationBelowThreshold : Real = 0.0;
            attribute shutdownTriggerDuration : Real = 300.0;
        }

        part alertSystem : AlertSystem {
            attribute alertActive : Boolean = false;
            attribute alertMessage : String = "Battery risk detected";
            attribute alertSeverity : SeverityLevel = SeverityLevel::HIGH;
            attribute notifyUser : Boolean = true;
            attribute notifyMaintenance : Boolean = true;
        }

        port batteryOut : BatteryDataPort;
        port riskOut : RiskDataPort;
    }

    part equipment : Equipment {
        attribute equipmentId : String = "DEVICE-001";
        attribute operationalStatus : Boolean = true;
    }

    connect equipment.batteryOut to equipment.riskOut;
}