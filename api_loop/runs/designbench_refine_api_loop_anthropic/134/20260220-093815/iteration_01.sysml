package BloodGlucoseMeterSafetySystem {
    private import ScalarValues::*;

    enum def BatteryStatus {
        enum NORMAL;
        enum LOW;
        enum DEPLETED;
        enum CHARGING_FAULT;
    }

    enum def AlarmLevel {
        enum INFO;
        enum WARNING;
        enum CRITICAL;
    }

    enum def MeasurementStatus {
        enum READY;
        enum IN_PROGRESS;
        enum COMPLETE;
        enum FAILED;
    }

    enum def ApprovalStatus {
        enum PENDING;
        enum APPROVED;
        enum REJECTED;
    }

    enum def RiskLevel {
        enum LOW_RISK;
        enum MEDIUM_RISK;
        enum HIGH_RISK;
    }

    part def BatteryMonitor {
        attribute currentStatus : BatteryStatus;
        attribute chargeLevel : Real;
        attribute lowBatteryThreshold : Real;
        attribute criticalBatteryThreshold : Real;
        attribute isChargingFault : Boolean;
        attribute detectionEnabled : Boolean;
    }

    part def AlarmSystem {
        attribute alarmLevel : AlarmLevel;
        attribute alarmActive : Boolean;
        attribute batteryAlarmEnabled : Boolean;
        attribute measurementAlarmEnabled : Boolean;
        attribute alertMessage : String;
    }

    part def EmergencyResponseUnit {
        attribute emergencyActive : Boolean;
        attribute responseTriggered : Boolean;
        attribute notifyUser : Boolean;
        attribute notifyCaregiver : Boolean;
        attribute treatmentDelayPrevented : Boolean;
    }

    part def BloodGlucoseSensor {
        attribute measurementStatus : MeasurementStatus;
        attribute glucoseReading : Real;
        attribute measurementValid : Boolean;
        attribute sensorFunctional : Boolean;
    }

    part def PreventiveMeasureController {
        attribute batteryPreventionActive : Boolean;
        attribute autoDetectionEnabled : Boolean;
        attribute preventiveMeasureApplied : Boolean;
        attribute riskLevel : RiskLevel;
    }

    part def RequirementsApprovalRecord {
        attribute requirementId : String;
        attribute approvalStatus : ApprovalStatus;
        attribute approvedBy : String;
        attribute approvalDate : String;
        attribute safetyRequirementMet : Boolean;
    }

    part def RiskManagementRecord {
        attribute riskScenarioId : String;
        attribute riskLevel : RiskLevel;
        attribute managementMeasureApplied : Boolean;
        attribute scenarioDescription : String;
        attribute mitigationVerified : Boolean;
    }

    part def UserInterface {
        attribute batteryWarningDisplayed : Boolean;
        attribute measurementResultDisplayed : Boolean;
        attribute emergencyAlertDisplayed : Boolean;
        attribute userNotified : Boolean;
    }

    part def TherapeuticSupportManager {
        attribute continuousSupportEnabled : Boolean;
        attribute patientSafetyEnsured : Boolean;
        attribute timelyMeasurementEnabled : Boolean;
        attribute safeOperationVerified : Boolean;
    }

    port def BatteryStatusPort {
        attribute statusOut : BatteryStatus;
        attribute chargeLevelOut : Real;
    }

    port def AlarmPort {
        attribute alarmLevelOut : AlarmLevel;
        attribute alarmActiveOut : Boolean;
    }

    port def MeasurementPort {
        attribute measurementStatusOut : MeasurementStatus;
        attribute glucoseValueOut : Real;
    }

    port def EmergencyPort {
        attribute emergencyActiveOut : Boolean;
        attribute responseOut : Boolean;
    }

    part def BloodGlucoseMeterSystem {
        attribute systemId : String;
        attribute systemOperational : Boolean;
        attribute safetyApproved : Boolean;
        attribute overallRiskAcceptable : Boolean;

        part batteryMonitor : BatteryMonitor;
        part alarmSystem : AlarmSystem;
        part emergencyUnit : EmergencyResponseUnit;
        part glucoseSensor : BloodGlucoseSensor;
        part preventiveController : PreventiveMeasureController;
        part approvalRecord : RequirementsApprovalRecord;
        part riskRecord : RiskManagementRecord;
        part userInterface : UserInterface;
        part therapeuticManager : TherapeuticSupportManager;

        port batteryStatusPort : BatteryStatusPort;
        port alarmPort : AlarmPort;
        port measurementPort : MeasurementPort;
        port emergencyPort : EmergencyPort;

        connect batteryMonitor to alarmSystem;
        connect batteryMonitor to preventiveController;
        connect alarmSystem to userInterface;
        connect alarmSystem to emergencyUnit;
        connect glucoseSensor to alarmSystem;
        connect glucoseSensor to userInterface;
        connect emergencyUnit to userInterface;
        connect emergencyUnit to therapeuticManager;
        connect preventiveController to alarmSystem;
        connect approvalRecord to therapeuticManager;
        connect riskRecord to preventiveController;
        connect riskRecord to therapeuticManager;
    }

    item def BatteryStatusSignal {
        attribute status : BatteryStatus;
        attribute chargeLevel : Real;
    }

    item def AlarmNotification {
        attribute level : AlarmLevel;
        attribute message : String;
        attribute timestamp : String;
    }

    item def MeasurementResult {
        attribute status : MeasurementStatus;
        attribute glucoseValue : Real;
        attribute isValid : Boolean;
    }

    item def EmergencyAlert {
        attribute alertLevel : AlarmLevel;
        attribute description : String;
        attribute actionRequired : Boolean;
    }

    item def RiskMitigationRecord {
        attribute scenarioId : String;
        attribute riskLevel : RiskLevel;
        attribute mitigationApplied : Boolean;
    }
}