package VehicleModularPlatform {
    private import ScalarValues::*;

    enum def VariantKind {
        enum VariantA;
        enum VariantB;
    }

    enum def ConfigKind {
        enum ConfigA;
        enum ConfigB;
    }

    port def PortP1 {
        attribute signal1 : String;
    }

    port def PortP2 {
        attribute signal2 : String;
    }

    port def SubsystemPortA {
        attribute linkId : Integer;
    }

    port def SubsystemPortB {
        attribute linkId : Integer;
    }

    part def Part1 {
        attribute componentId : Integer;
        attribute isActive : Boolean;
    }

    part def Part2 {
        attribute componentId : Integer;
        attribute weight : Real;
    }

    part def Part3 {
        attribute componentId : Integer;
        port p1 : PortP1;
    }

    part def Part4 {
        attribute componentId : Integer;
        attribute capacity : Real;
    }

    part def Part5 {
        attribute componentId : Integer;
        attribute variant : VariantKind;
        port p2 : PortP2;
    }

    part def Part6 {
        attribute componentId : Integer;
        attribute rating : Real;
    }

    part def SubsystemA {
        attribute subsystemId : Integer;
        attribute config : ConfigKind;
        port outPortA : SubsystemPortA;
        part compPart1 : Part1;
        part compPart2 : Part2;
        part compPart3 : Part3;
    }

    part def SubsystemAVariant {
        attribute subsystemId : Integer;
        attribute config : ConfigKind;
        port outPortA : SubsystemPortA;
        part compPart1 : Part1;
        part compPart4 : Part4;
        part compPart3 : Part3;
    }

    part def SubsystemB {
        attribute subsystemId : Integer;
        attribute config : ConfigKind;
        port inPortB : SubsystemPortB;
        part compPart5 : Part5;
        part compPart6 : Part6;
    }

    part def SubsystemBVariant {
        attribute subsystemId : Integer;
        attribute config : ConfigKind;
        port inPortB : SubsystemPortB;
        part compPart5a : Part5;
        part compPart5b : Part5;
        part compPart6 : Part6;
    }

    part def VehicleConfigA {
        attribute configId : Integer;
        attribute configType : ConfigKind = ConfigKind::ConfigA;
        attribute maxLoad : Real;
        attribute isValidConfig : Boolean;
        part subsysA : SubsystemA;
        part subsysB : SubsystemB;
        connect subsysA.outPortA to subsysB.inPortB;
    }

    part def VehicleConfigB {
        attribute configId : Integer;
        attribute configType : ConfigKind = ConfigKind::ConfigB;
        attribute maxLoad : Real;
        attribute isValidConfig : Boolean;
        part subsysAv : SubsystemAVariant;
        part subsysBv : SubsystemBVariant;
        connect subsysAv.outPortA to subsysBv.inPortB;
    }

    part vehiclePlatform : VehicleConfigA {
        attribute configId = 1;
        attribute maxLoad = 5000.0;
        attribute isValidConfig = true;
        part :>> subsysA {
            attribute subsystemId = 101;
            attribute config = ConfigKind::ConfigA;
            part :>> compPart1 {
                attribute componentId = 1;
                attribute isActive = true;
            }
            part :>> compPart2 {
                attribute componentId = 2;
                attribute weight = 12.5;
            }
            part :>> compPart3 {
                attribute componentId = 3;
            }
        }
        part :>> subsysB {
            attribute subsystemId = 201;
            attribute config = ConfigKind::ConfigA;
            part :>> compPart5 {
                attribute componentId = 5;
                attribute variant = VariantKind::VariantA;
            }
            part :>> compPart6 {
                attribute componentId = 6;
                attribute rating = 9.8;
            }
        }
    }

    part vehiclePlatformB : VehicleConfigB {
        attribute configId = 2;
        attribute maxLoad = 8000.0;
        attribute isValidConfig = true;
        part :>> subsysAv {
            attribute subsystemId = 102;
            attribute config = ConfigKind::ConfigB;
            part :>> compPart1 {
                attribute componentId = 11;
                attribute isActive = true;
            }
            part :>> compPart4 {
                attribute componentId = 14;
                attribute capacity = 200.0;
            }
            part :>> compPart3 {
                attribute componentId = 13;
            }
        }
        part :>> subsysBv {
            attribute subsystemId = 202;
            attribute config = ConfigKind::ConfigB;
            part :>> compPart5a {
                attribute componentId = 51;
                attribute variant = VariantKind::VariantA;
            }
            part :>> compPart5b {
                attribute componentId = 52;
                attribute variant = VariantKind::VariantB;
            }
            part :>> compPart6 {
                attribute componentId = 61;
                attribute rating = 7.5;
            }
        }
    }
}