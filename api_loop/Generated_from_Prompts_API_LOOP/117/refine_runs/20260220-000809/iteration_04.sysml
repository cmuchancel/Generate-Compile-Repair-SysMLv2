package VehicleProductLineConfiguration {
  private import ScalarValues::*;

  enum def PerformanceLevel {
    enum highPerf;
    enum standardPerf;
  }

  enum def EngineType {
    enum sixCylinder;
    enum fourCylinder;
  }

  enum def TransmissionType {
    enum automatic;
    enum manual;
  }

  enum def WheelType {
    enum wideRim;
    enum narrowRim;
  }

  enum def VehicleVariantType {
    enum sedan;
    enum suv;
    enum truck;
  }

  enum def UserPreferenceType {
    enum performanceFocused;
    enum balanced;
    enum economyFocused;
  }

  part def Engine {
    attribute engineType : EngineType;
  }

  part def Transmission {
    attribute transmissionType : TransmissionType;
  }

  part def WheelSet {
    attribute wheelType : WheelType;
  }

  part def SixCylinderEngine : Engine {
    attribute engineType : EngineType = EngineType::sixCylinder;
  }

  part def FourCylinderEngine : Engine {
    attribute engineType : EngineType = EngineType::fourCylinder;
  }

  part def AutomaticTransmission : Transmission {
    attribute transmissionType : TransmissionType = TransmissionType::automatic;
  }

  part def ManualTransmission : Transmission {
    attribute transmissionType : TransmissionType = TransmissionType::manual;
  }

  part def WideRimWheelSet : WheelSet {
    attribute wheelType : WheelType = WheelType::wideRim;
  }

  part def NarrowRimWheelSet : WheelSet {
    attribute wheelType : WheelType = WheelType::narrowRim;
  }

  part def VehicleConfiguration {
    attribute performanceRequirement : PerformanceLevel;
    attribute userPreference : UserPreferenceType;
    attribute modelVariant : VehicleVariantType;
    part engine : Engine;
    part transmission : Transmission;
    part wheels : WheelSet;
  }

  part def HighPerformanceVariant : VehicleConfiguration {
    attribute performanceRequirement : PerformanceLevel = PerformanceLevel::highPerf;
    attribute userPreference : UserPreferenceType = UserPreferenceType::performanceFocused;
    part engine : SixCylinderEngine;
    part transmission : AutomaticTransmission;
    part wheels : WideRimWheelSet;
  }

  part def StandardPerformanceVariant : VehicleConfiguration {
    attribute performanceRequirement : PerformanceLevel = PerformanceLevel::standardPerf;
    attribute userPreference : UserPreferenceType = UserPreferenceType::balanced;
    part engine : FourCylinderEngine;
    part transmission : ManualTransmission;
    part wheels : NarrowRimWheelSet;
  }

  part def HighPerformanceSUV : HighPerformanceVariant {
    attribute modelVariant : VehicleVariantType = VehicleVariantType::suv;
  }

  part def HighPerformanceSedan : HighPerformanceVariant {
    attribute modelVariant : VehicleVariantType = VehicleVariantType::sedan;
  }

  part def StandardSedan : StandardPerformanceVariant {
    attribute modelVariant : VehicleVariantType = VehicleVariantType::sedan;
  }

  part def StandardTruck : StandardPerformanceVariant {
    attribute modelVariant : VehicleVariantType = VehicleVariantType::truck;
  }

  part def VehicleProductLineSystem {
    part highPerformanceSUV : HighPerformanceSUV;
    part highPerformanceSedan : HighPerformanceSedan;
    part standardSedan : StandardSedan;
    part standardTruck : StandardTruck;
  }
}