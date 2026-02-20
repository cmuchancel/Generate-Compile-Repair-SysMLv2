package VehiclePlatform {
  private import ScalarValues::*;

  enum def BehaviorVariant {
    enum VariantX;
    enum VariantY;
  }

  enum def ConfigVariant {
    enum ConfigA;
    enum ConfigB;
  }

  item def Signal;

  port def SignalPort { }

  part def Part1 {
    attribute id : Integer;
  }

  part def Part2 {
    attribute id : Integer;
  }

  part def Part3 {
    port p1 : SignalPort;
  }

  part def Part4 {
    attribute id : Integer;
  }

  part def Part5 {
    port p2 : SignalPort;
    attribute behavior : BehaviorVariant = BehaviorVariant::VariantX;
  }

  part def Part6 {
    attribute id : Integer;
  }

  part def SubsystemA {
    port aPort : SignalPort;
    part p1a : Part1;
    part p3a : Part3;
    part p5a : Part5;
  }

  part def SubsystemB {
    port bPort : SignalPort;
    part p2a : Part2;
    part p4a : Part4;
    part p6a : Part6;
  }

  part def VehicleConfigA {
    attribute config : ConfigVariant = ConfigVariant::ConfigA;
    part subsA : SubsystemA;
    part subsB : SubsystemB;
    connect subsA.aPort to subsB.bPort;
    connect subsA.p3a.p1 to subsB.bPort;
    connect subsA.p5a.p2 to subsB.bPort;
  }

  part def VehicleConfigB {
    attribute config : ConfigVariant = ConfigVariant::ConfigB;
    part subsA1 : SubsystemA;
    part subsA2 : SubsystemA;
    part subsB1 : SubsystemB;
    part extraPart4 : Part4;
    part part5b : Part5 {
      attribute behavior : BehaviorVariant = BehaviorVariant::VariantY;
    }
    connect subsA1.aPort to subsB1.bPort;
    connect subsA2.aPort to subsB1.bPort;
    connect part5b.p2 to subsA2.aPort;
    connect subsA1.p5a.p2 to subsB1.bPort;
  }
}