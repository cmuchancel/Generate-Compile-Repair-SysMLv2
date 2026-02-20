package AutomotiveSystem {
    private import ScalarValues::*;

    enum def SafetyStatus {
        enum MANDATORY;
        enum OPTIONAL;
        enum NON_SAFETY;
    }

    part def RootSystem;
    part def ChassisSubsystem;
    part def PowertrainSubsystem;
    part def SafetySubsystem;
    part def ElectricalSubsystem;

    part def SeatBelt;
    part def Bumper;
    part def DriverAirbag;
    part def AntiLockBrakingSystem;
    part def Engine;
    part def Transmission;
    part def FuelSystem;
    part def Battery;
    part def AlternatorUnit;
    part def WiringHarness;
    part def SuspensionSystem;
    part def SteeringSystem;
    part def BrakeSystem;

    port def MechanicalPort;
    port def ElectricalPort;
    port def SignalPort;

    part root : RootSystem {

        part chassis : ChassisSubsystem {
            attribute subsystemName : String = "Chassis Subsystem";
            attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;

            port chassisMechOut : MechanicalPort;
            port chassisElecIn : ElectricalPort;

            part suspension : SuspensionSystem {
                attribute componentName : String = "Suspension System";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port mechPort : MechanicalPort;
            }

            part steering : SteeringSystem {
                attribute componentName : String = "Steering System";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port mechPort : MechanicalPort;
            }

            part brakes : BrakeSystem {
                attribute componentName : String = "Brake System";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port mechPort : MechanicalPort;
                port sigPort : SignalPort;
            }
        }

        part powertrain : PowertrainSubsystem {
            attribute subsystemName : String = "Powertrain Subsystem";
            attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;

            port powerOut : MechanicalPort;
            port elecIn : ElectricalPort;

            part engine : Engine {
                attribute componentName : String = "Engine";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port mechOut : MechanicalPort;
                port elecIn : ElectricalPort;
            }

            part transmission : Transmission {
                attribute componentName : String = "Transmission";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port mechIn : MechanicalPort;
                port mechOut : MechanicalPort;
            }

            part fuelSystem : FuelSystem {
                attribute componentName : String = "Fuel System";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port elecIn : ElectricalPort;
            }

            connect engine.mechOut to transmission.mechIn;
        }

        part safetySubsys : SafetySubsystem {
            attribute subsystemName : String = "Safety Subsystem";
            attribute safetyClassification : SafetyStatus = SafetyStatus::MANDATORY;

            port sigIn : SignalPort;

            part seatBelt : SeatBelt {
                attribute componentName : String = "Seat Belt";
                attribute safetyClassification : SafetyStatus = SafetyStatus::MANDATORY;
                attribute isMandatory : Boolean = true;
                attribute isEquipped : Boolean = true;
            }

            part bumper : Bumper {
                attribute componentName : String = "Bumper";
                attribute safetyClassification : SafetyStatus = SafetyStatus::MANDATORY;
                attribute isMandatory : Boolean = true;
                attribute isEquipped : Boolean = true;
            }

            part driverAirbag : DriverAirbag {
                attribute componentName : String = "Driver Airbag";
                attribute safetyClassification : SafetyStatus = SafetyStatus::OPTIONAL;
                attribute isMandatory : Boolean = false;
                attribute isEquipped : Boolean = true;
                port sigIn : SignalPort;
            }

            part abs : AntiLockBrakingSystem {
                attribute componentName : String = "Anti-Lock Braking System";
                attribute safetyClassification : SafetyStatus = SafetyStatus::OPTIONAL;
                attribute isMandatory : Boolean = false;
                attribute isEquipped : Boolean = true;
                port sigIn : SignalPort;
                port mechOut : MechanicalPort;
            }
        }

        part electrical : ElectricalSubsystem {
            attribute subsystemName : String = "Electrical Subsystem";
            attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;

            port elecOut : ElectricalPort;
            port sigOut : SignalPort;

            part battery : Battery {
                attribute componentName : String = "Battery";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port elecOut : ElectricalPort;
            }

            part alternator : AlternatorUnit {
                attribute componentName : String = "Alternator";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port mechIn : MechanicalPort;
                port elecOut : ElectricalPort;
            }

            part wiring : WiringHarness {
                attribute componentName : String = "Wiring Harness";
                attribute safetyClassification : SafetyStatus = SafetyStatus::NON_SAFETY;
                port elecIn : ElectricalPort;
                port elecOut : ElectricalPort;
                port sigOut : SignalPort;
            }

            connect battery.elecOut to wiring.elecIn;
        }

        connect electrical.elecOut to powertrain.elecIn;
        connect electrical.elecOut to chassis.chassisElecIn;
        connect electrical.sigOut to safetySubsys.sigIn;
        connect chassis.brakes.sigPort to safetySubsys.abs.sigIn;
    }
}