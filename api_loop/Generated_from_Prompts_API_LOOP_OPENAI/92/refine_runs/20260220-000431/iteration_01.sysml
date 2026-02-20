package SystemArchitecture {
  private import ScalarValues::*;

  enum def CapabilityKind {
    enum StructuralControl;
    enum PowerManagement;
    enum CommunicationHandling;
    enum SensingActuation;
  }

  enum def FunctionKind {
    enum Monitor;
    enum Compute;
    enum Distribute;
    enum Interface;
  }

  item def DataPacket;
  item def PowerFlow;
  item def ControlSignal;
  item def StatusReport;

  port def DataPort {
    item data : DataPacket;
  }

  port def PowerPort {
    item power : PowerFlow;
  }

  port def ControlPort {
    item control : ControlSignal;
  }

  port def StatusPort {
    item status : StatusReport;
  }

  part def SensorUnit {
    attribute capability : CapabilityKind = SensingActuation;
    attribute function : FunctionKind = Monitor;
    port dataOut : DataPort;
    port statusOut : StatusPort;
  }

  part def ActuatorUnit {
    attribute capability : CapabilityKind = SensingActuation;
    attribute function : FunctionKind = Interface;
    port controlIn : ControlPort;
    port statusOut : StatusPort;
  }

  part def ControllerUnit {
    attribute capability : CapabilityKind = StructuralControl;
    attribute function : FunctionKind = Compute;
    port dataIn : DataPort;
    port controlOut : ControlPort;
    port statusOut : StatusPort;
  }

  part def ControlSubsystem {
    attribute capability : CapabilityKind = StructuralControl;
    attribute function : FunctionKind = Compute;
    part sensor : SensorUnit;
    part controller : ControllerUnit;
    part actuator : ActuatorUnit;
    port dataIn : DataPort;
    port statusOut : StatusPort;
    connect sensor.dataOut to controller.dataIn;
    connect controller.controlOut to actuator.controlIn;
    connect controller.statusOut to statusOut;
    connect dataIn to controller.dataIn;
  }

  part def PowerSourceUnit {
    attribute capability : CapabilityKind = PowerManagement;
    attribute function : FunctionKind = Distribute;
    port powerOut : PowerPort;
    port statusOut : StatusPort;
  }

  part def PowerDistributionUnit {
    attribute capability : CapabilityKind = PowerManagement;
    attribute function : FunctionKind = Distribute;
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port statusOut : StatusPort;
  }

  part def PowerSubsystem {
    attribute capability : CapabilityKind = PowerManagement;
    attribute function : FunctionKind = Distribute;
    part source : PowerSourceUnit;
    part distributor : PowerDistributionUnit;
    port powerOut : PowerPort;
    port statusOut : StatusPort;
    connect source.powerOut to distributor.powerIn;
    connect distributor.powerOut to powerOut;
    connect distributor.statusOut to statusOut;
  }

  part def CommunicationUnit {
    attribute capability : CapabilityKind = CommunicationHandling;
    attribute function : FunctionKind = Interface;
    port dataIn : DataPort;
    port dataOut : DataPort;
    port statusOut : StatusPort;
  }

  part def CommunicationSubsystem {
    attribute capability : CapabilityKind = CommunicationHandling;
    attribute function : FunctionKind = Interface;
    part comms : CommunicationUnit;
    port externalDataIn : DataPort;
    port internalDataOut : DataPort;
    port statusOut : StatusPort;
    connect externalDataIn to comms.dataIn;
    connect comms.dataOut to internalDataOut;
    connect comms.statusOut to statusOut;
  }

  part def System {
    attribute capability : CapabilityKind = StructuralControl;
    attribute function : FunctionKind = Interface;
    part controlSubsystem : ControlSubsystem;
    part powerSubsystem : PowerSubsystem;
    part communicationSubsystem : CommunicationSubsystem;
    port externalDataIn : DataPort;
    port externalPowerIn : PowerPort;
    port systemStatusOut : StatusPort;
    connect externalDataIn to communicationSubsystem.externalDataIn;
    connect communicationSubsystem.internalDataOut to controlSubsystem.dataIn;
    connect externalPowerIn to powerSubsystem.source.powerOut;
    connect powerSubsystem.powerOut to controlSubsystem.actuator.controlIn;
    connect controlSubsystem.statusOut to systemStatusOut;
    connect powerSubsystem.statusOut to systemStatusOut;
    connect communicationSubsystem.statusOut to systemStatusOut;
  }
}