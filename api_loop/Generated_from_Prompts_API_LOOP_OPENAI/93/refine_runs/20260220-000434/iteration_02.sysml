package AutomotiveSystemModel {
  private import ScalarValues::*;

  enum def SafetyStatus { Mandatory; Optional; }

  port def GenericPort;

  part def Component { 
    port ifc : GenericPort; 
  }

  part def SafetyComponent : Component { 
    attribute isSafety : Boolean = true; 
  }

  part def NonSafetyComponent : Component { 
    attribute isSafety : Boolean = false; 
  }

  part def SeatBelt : SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Mandatory; 
  }

  part def Bumper : SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Mandatory; 
  }

  part def DriverAirbag : SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Optional; 
  }

  part def ABS : SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Optional; 
  }

  part def Engine : NonSafetyComponent;
  part def Transmission : NonSafetyComponent;
  part def Battery : NonSafetyComponent;
  part def Alternator : NonSafetyComponent;
  part def HeadUnit : NonSafetyComponent;
  part def Speakers : NonSafetyComponent;
  part def Chassis : NonSafetyComponent;
  part def Doors : NonSafetyComponent;
  part def BaseBrakes : NonSafetyComponent;

  part def BodySubsystem : NonSafetyComponent {
    part bumper : Bumper;
    part chassis : Chassis;
    part doors : Doors;
  }

  part def RestraintSubsystem : NonSafetyComponent {
    part seatBelt : SeatBelt;
    part driverAirbag : DriverAirbag;
  }

  part def BrakingSubsystem : NonSafetyComponent {
    part baseBrakes : BaseBrakes;
    part abs : ABS;
  }

  part def PowertrainSubsystem : NonSafetyComponent {
    part engine : Engine;
    part transmission : Transmission;
  }

  part def ElectricalSubsystem : NonSafetyComponent {
    part battery : Battery;
    part alternator : Alternator;
  }

  part def InfotainmentSubsystem : NonSafetyComponent {
    part headUnit : HeadUnit;
    part speakers : Speakers;
  }

  part def AutomotiveSystem : NonSafetyComponent {
    part body : BodySubsystem;
    part restraint : RestraintSubsystem;
    part braking : BrakingSubsystem;
    part powertrain : PowertrainSubsystem;
    part electrical : ElectricalSubsystem;
    part infotainment : InfotainmentSubsystem;
  }

  part def SystemTreeView : NonSafetyComponent {
    part root : AutomotiveSystem;
  }

  part def SafetyComponentsTable : NonSafetyComponent {
    part seatBeltEntry : SeatBelt;
    part bumperEntry : Bumper;
    part driverAirbagEntry : DriverAirbag;
    part absEntry : ABS;
  }

  part def NonSafetyComponentsTable : NonSafetyComponent {
    part automotiveSystemEntry : AutomotiveSystem;
    part bodySubsystemEntry : BodySubsystem;
    part restraintSubsystemEntry : RestraintSubsystem;
    part brakingSubsystemEntry : BrakingSubsystem;
    part powertrainSubsystemEntry : PowertrainSubsystem;
    part electricalSubsystemEntry : ElectricalSubsystem;
    part infotainmentSubsystemEntry : InfotainmentSubsystem;
    part engineEntry : Engine;
    part transmissionEntry : Transmission;
    part batteryEntry : Battery;
    part alternatorEntry : Alternator;
    part headUnitEntry : HeadUnit;
    part speakersEntry : Speakers;
    part chassisEntry : Chassis;
    part doorsEntry : Doors;
    part baseBrakesEntry : BaseBrakes;
  }

  part def SystemViews : NonSafetyComponent {
    part treeView : SystemTreeView;
    part safetyTable : SafetyComponentsTable;
    part nonSafetyTable : NonSafetyComponentsTable;
  }

  part automotiveSystem : AutomotiveSystem;
  part views : SystemViews;
}