package AutomotiveSystemModel {
  private import ScalarValues::*;

  enum def SafetyStatus { Mandatory; Optional; }

  port def GenericPort;

  part def Component { 
    port ifc : GenericPort; 
  }

  part def SafetyComponent specializes Component { 
    attribute isSafety : Boolean = true; 
  }

  part def NonSafetyComponent specializes Component { 
    attribute isSafety : Boolean = false; 
  }

  part def SeatBelt specializes SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Mandatory; 
  }

  part def Bumper specializes SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Mandatory; 
  }

  part def DriverAirbag specializes SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Optional; 
  }

  part def ABS specializes SafetyComponent { 
    attribute safetyStatus : SafetyStatus = SafetyStatus::Optional; 
  }

  part def Engine specializes NonSafetyComponent { }
  part def Transmission specializes NonSafetyComponent { }
  part def Battery specializes NonSafetyComponent { }
  part def Alternator specializes NonSafetyComponent { }
  part def HeadUnit specializes NonSafetyComponent { }
  part def Speakers specializes NonSafetyComponent { }
  part def Chassis specializes NonSafetyComponent { }
  part def Doors specializes NonSafetyComponent { }
  part def BaseBrakes specializes NonSafetyComponent { }

  part def BodySubsystem specializes NonSafetyComponent {
    part bumper : Bumper;
    part chassis : Chassis;
    part doors : Doors;
  }

  part def RestraintSubsystem specializes NonSafetyComponent {
    part seatBelt : SeatBelt;
    part driverAirbag : DriverAirbag;
  }

  part def BrakingSubsystem specializes NonSafetyComponent {
    part baseBrakes : BaseBrakes;
    part abs : ABS;
  }

  part def PowertrainSubsystem specializes NonSafetyComponent {
    part engine : Engine;
    part transmission : Transmission;
  }

  part def ElectricalSubsystem specializes NonSafetyComponent {
    part battery : Battery;
    part alternator : Alternator;
  }

  part def InfotainmentSubsystem specializes NonSafetyComponent {
    part headUnit : HeadUnit;
    part speakers : Speakers;
  }

  part def AutomotiveSystem specializes NonSafetyComponent {
    part body : BodySubsystem;
    part restraint : RestraintSubsystem;
    part braking : BrakingSubsystem;
    part powertrain : PowertrainSubsystem;
    part electrical : ElectricalSubsystem;
    part infotainment : InfotainmentSubsystem;
  }

  part def SystemTreeView specializes NonSafetyComponent {
    part root : AutomotiveSystem;
  }

  part def SafetyComponentsTable specializes NonSafetyComponent {
    part seatBeltEntry : SeatBelt;
    part bumperEntry : Bumper;
    part driverAirbagEntry : DriverAirbag;
    part absEntry : ABS;
  }

  part def NonSafetyComponentsTable specializes NonSafetyComponent {
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

  part def SystemViews specializes NonSafetyComponent {
    part treeView : SystemTreeView;
    part safetyTable : SafetyComponentsTable;
    part nonSafetyTable : NonSafetyComponentsTable;
  }

  part automotiveSystem : AutomotiveSystem;
  part views : SystemViews;
}