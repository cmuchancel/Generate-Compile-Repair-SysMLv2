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
    ref item data : DataPacket;
  }

  port def PowerPort {
    ref item power : PowerFlow;
  }

  port def ControlPort {
    ref item control : ControlSignal;
  }

  port def StatusPort {
    ref item status : StatusReport;
  }

  part def SensorUnit {
    attribute capability : CapabilityKind = CapabilityKind::SensingActuation;
    attribute function : FunctionKind = FunctionKind::Monitor;
    port dataOut : DataPort;
    port statusOut : StatusPort;
  }

  part def ActuatorUnit {
    attribute capability : CapabilityKind = CapabilityKind::SensingActuation;
    attribute function : FunctionKind = FunctionKind::Interface;
    port controlIn : ControlPort;
    port powerIn : PowerPort;
    port statusOut : StatusPort;
  }

  part def ControllerUnit {
    attribute capability : CapabilityKind = CapabilityKind::StructuralControl;
    attribute function : FunctionKind = FunctionKind::Compute;
    port dataIn : DataPort;
    port controlOut : ControlPort;
    port statusOut : StatusPort;
  }

  part def ControlSubsystem {
    attribute capability : CapabilityKind = CapabilityKind::StructuralControl;
    attribute function : FunctionKind = FunctionKind::Compute;
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
    attribute capability : CapabilityKind = CapabilityKind::PowerManagement;
    attribute function : FunctionKind = FunctionKind::Distribute;
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port statusOut : StatusPort;
  }

  part def PowerDistributionUnit {
    attribute capability : CapabilityKind = CapabilityKind::PowerManagement;
    attribute function : FunctionKind = FunctionKind::Distribute;
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port statusOut : StatusPort;
  }

  part def PowerSubsystem {
    attribute capability : CapabilityKind = CapabilityKind::PowerManagement;
    attribute function : FunctionKind = FunctionKind::Distribute;
    part source : PowerSourceUnit;
    part distributor : PowerDistributionUnit;
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port statusOut : StatusPort;
    connect powerIn to source.powerIn;
    connect source.powerOut to distributor.powerIn;
    connect distributor.powerOut to powerOut;
    connect distributor.statusOut to statusOut;
  }

  part def CommunicationUnit {
    attribute capability : CapabilityKind = CapabilityKind::CommunicationHandling;
    attribute function : FunctionKind = FunctionKind::Interface;
    port dataIn : DataPort;
    port dataOut : DataPort;
    port statusOut : StatusPort;
  }

  part def CommunicationSubsystem {
    attribute capability : CapabilityKind = CapabilityKind::CommunicationHandling;
    attribute function : FunctionKind = FunctionKind::Interface;
    part comms : CommunicationUnit;
    port externalDataIn : DataPort;
    port internalDataOut : DataPort;
    port statusOut : StatusPort;
    connect externalDataIn to comms.dataIn;
    connect comms.dataOut to internalDataOut;
    connect comms.statusOut to statusOut;
  }

  part def System {
    attribute capability : CapabilityKind = CapabilityKind::StructuralControl;
    attribute function : FunctionKind = FunctionKind::Interface;
    part controlSubsystem : ControlSubsystem;
    part powerSubsystem : PowerSubsystem;
    part communicationSubsystem : CommunicationSubsystem;
    port externalDataIn : DataPort;
    port externalPowerIn : PowerPort;
    port systemStatusOut : StatusPort;
    connect externalDataIn to communicationSubsystem.externalDataIn;
    connect communicationSubsystem.internalDataOut to controlSubsystem.dataIn;
    connect externalPowerIn to powerSubsystem.powerIn;
    connect powerSubsystem.powerOut to controlSubsystem.actuator.powerIn;
    connect controlSubsystem.statusOut to systemStatusOut;
    connect powerSubsystem.statusOut to systemStatusOut;
    connect communicationSubsystem.statusOut to systemStatusOut;
  }
}