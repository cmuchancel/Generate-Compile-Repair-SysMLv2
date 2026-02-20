package SystemDataModel {
  private import ScalarValues::*;

  item def UnsignedInteger {
    attribute value : Integer = 0;
    attribute minValue : Integer = 0;
  }

  item def UnsignedFloat {
    attribute value : Real = 0.0;
    attribute minValue : Real = 0.0;
  }

  item def DateTimeValue {
    attribute value : DateTime;
  }

  item def StringValue {
    attribute value : String;
  }

  item def BooleanValue {
    attribute value : Boolean;
  }

  enum def StatusLevel {
    enum critical;
    enum warning;
    enum normal;
  }

  enum def StatusColor {
    enum red;
    enum yellow;
    enum green;
  }

  part def StatusColorTable {
    attribute criticalColor : StatusColor = StatusColor::red;
    attribute warningColor : StatusColor = StatusColor::yellow;
    attribute normalColor : StatusColor = StatusColor::green;
  }

  part def StatusDefinition {
    attribute level : StatusLevel;
    attribute color : StatusColor;
  }

  enum def DiameterOption {
    enum small;
    enum medium;
    enum large;
  }

  part def DiameterOptionValue {
    attribute option : DiameterOption;
    attribute diameterMM : Real;
  }

  part def DiameterCatalog {
    part small : DiameterOptionValue {
      attribute option : DiameterOption = DiameterOption::small;
      attribute diameterMM : Real = 60.0;
    }
    part medium : DiameterOptionValue {
      attribute option : DiameterOption = DiameterOption::medium;
      attribute diameterMM : Real = 70.0;
    }
    part large : DiameterOptionValue {
      attribute option : DiameterOption = DiameterOption::large;
      attribute diameterMM : Real = 80.0;
    }
  }

  part def DiameterSelection {
    attribute selectedOption : DiameterOption = DiameterOption::small;
    attribute selectedDiameterMM : Real = 60.0;
  }

  part def DataTypeLibrary {
    part statusColors : StatusColorTable;
    part diameters : DiameterCatalog;
    part diameterSelection : DiameterSelection;
  }
}