package VehicleProductLineConfiguration {
  private import ScalarValues::*;

  enum def PerformanceLevel {
    enum high;
    enum standard;
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

  part def VehicleConfiguration {
    attribute performanceRequirement : PerformanceLevel;
    attribute userPreference : UserPreferenceType;
    attribute variant : VehicleVariantType;
    part engine : Engine;
    part transmission : Transmission;
    part wheels : WheelSet;
  }

  part def HighPerformanceVariant {
    attribute performanceRequirement : PerformanceLevel = PerformanceLevel::high;
    attribute userPreference : UserPreferenceType = UserPreferenceType::performanceFocused;
    attribute variant : VehicleVariantType;
    part engine : Engine {
      attribute engineType : EngineType = EngineType::sixCylinder;
    }
    part transmission : Transmission {
      attribute transmissionType : TransmissionType = TransmissionType::automatic;
    }
    part wheels : WheelSet {
      attribute wheelType : WheelType = WheelType::wideRim;
    }
  }

  part def StandardPerformanceVariant {
    attribute performanceRequirement : PerformanceLevel = PerformanceLevel::standard;
    attribute userPreference : UserPreferenceType = UserPreferenceType::balanced;
    attribute variant : VehicleVariantType;
    part engine : Engine {
      attribute engineType : EngineType = EngineType::fourCylinder;
    }
    part transmission : Transmission {
      attribute transmissionType : TransmissionType = TransmissionType::manual;
    }
    part wheels : WheelSet {
      attribute wheelType : WheelType = WheelType::narrowRim;
    }
  }

  part def VehicleProductLineSystem {
    part highPerformanceSUV : HighPerformanceVariant {
      attribute variant : VehicleVariantType = VehicleVariantType::suv;
    }
    part highPerformanceSedan : HighPerformanceVariant {
      attribute variant : VehicleVariantType = VehicleVariantType::sedan;
    }
    part standardSedan : StandardPerformanceVariant {
      attribute variant : VehicleVariantType = VehicleVariantType::sedan;
    }
    part standardTruck : StandardPerformanceVariant {
      attribute variant : VehicleVariantType = VehicleVariantType::truck;
    }
  }
}