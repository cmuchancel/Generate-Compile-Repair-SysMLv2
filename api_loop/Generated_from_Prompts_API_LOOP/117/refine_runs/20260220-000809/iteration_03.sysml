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

  part def VehicleConfiguration {
    attribute performanceRequirement : PerformanceLevel;
    attribute userPreference : UserPreferenceType;
    attribute modelVariant : VehicleVariantType;
    part engine : Engine;
    part transmission : Transmission;
    part wheels : WheelSet;
  }

  part def HighPerformanceVariant {
    attribute performanceRequirement : PerformanceLevel = highPerf;
    attribute userPreference : UserPreferenceType = performanceFocused;
    attribute modelVariant : VehicleVariantType;
    part engine : Engine {
      attribute engineType : EngineType = sixCylinder;
    }
    part transmission : Transmission {
      attribute transmissionType : TransmissionType = automatic;
    }
    part wheels : WheelSet {
      attribute wheelType : WheelType = wideRim;
    }
  }

  part def StandardPerformanceVariant {
    attribute performanceRequirement : PerformanceLevel = standardPerf;
    attribute userPreference : UserPreferenceType = balanced;
    attribute modelVariant : VehicleVariantType;
    part engine : Engine {
      attribute engineType : EngineType = fourCylinder;
    }
    part transmission : Transmission {
      attribute transmissionType : TransmissionType = manual;
    }
    part wheels : WheelSet {
      attribute wheelType : WheelType = narrowRim;
    }
  }

  part def VehicleProductLineSystem {
    part highPerformanceSUV : HighPerformanceVariant {
      attribute modelVariant : VehicleVariantType = suv;
    }
    part highPerformanceSedan : HighPerformanceVariant {
      attribute modelVariant : VehicleVariantType = sedan;
    }
    part standardSedan : StandardPerformanceVariant {
      attribute modelVariant : VehicleVariantType = sedan;
    }
    part standardTruck : StandardPerformanceVariant {
      attribute modelVariant : VehicleVariantType = truck;
    }
  }
}