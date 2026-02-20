package VehicleSafetySecurityModel {
  private import ScalarValues::*;

  enum def FeatureType { safety; security; }
  enum def SafetyLevel { mandatory; optional; notApplicable; }
  enum def ViewType { tree; table; }

  part def VehicleComponent {
    attribute featureType : FeatureType;
    attribute safetyLevel : SafetyLevel;
  }

  part def SeatBelt {
    attribute featureType : FeatureType = FeatureType::safety;
    attribute safetyLevel : SafetyLevel = SafetyLevel::mandatory;
  }

  part def Bumper {
    attribute featureType : FeatureType = FeatureType::safety;
    attribute safetyLevel : SafetyLevel = SafetyLevel::mandatory;
  }

  part def Airbag {
    attribute featureType : FeatureType = FeatureType::safety;
    attribute safetyLevel : SafetyLevel = SafetyLevel::optional;
  }

  part def AntiLockBrakingSystem {
    attribute featureType : FeatureType = FeatureType::safety;
    attribute safetyLevel : SafetyLevel = SafetyLevel::optional;
  }

  part def AlarmDevice {
    attribute featureType : FeatureType = FeatureType::security;
    attribute safetyLevel : SafetyLevel = SafetyLevel::notApplicable;
  }

  part def KeylessEntrySystem {
    attribute featureType : FeatureType = FeatureType::security;
    attribute safetyLevel : SafetyLevel = SafetyLevel::notApplicable;
  }

  part def Vehicle {
    part seatBelt : SeatBelt;
    part bumper : Bumper;
    part airbag : Airbag;
    part abs : AntiLockBrakingSystem;
    part alarm : AlarmDevice;
    part keylessEntry : KeylessEntrySystem;
  }

  part def Filter {
    attribute filterType : FeatureType;
  }

  part def SafetyFilter {
    attribute filterType : FeatureType = FeatureType::safety;
  }

  part def SecurityFilter {
    attribute filterType : FeatureType = FeatureType::security;
  }

  part def Visualization {
    attribute viewType : ViewType;
  }

  part def TreeView {
    attribute viewType : ViewType = ViewType::tree;
  }

  part def TableView {
    attribute viewType : ViewType = ViewType::table;
  }

  part def SafetySecurityManagementSystem {
    part vehicle : Vehicle;
    part safetyFilter : SafetyFilter;
    part securityFilter : SecurityFilter;
    part treeView : TreeView;
    part tableView : TableView;
  }
}