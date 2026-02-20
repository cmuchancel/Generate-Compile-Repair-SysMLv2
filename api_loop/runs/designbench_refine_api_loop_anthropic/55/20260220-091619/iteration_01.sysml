package VehicleMassManagement {
    private import ScalarValues::*;

    attribute def MassValue {
        attribute value : Real;
    }

    part def Component {
        attribute serialNumber : String;
        attribute mass : Real;
    }

    part def Engine specializes Component {
        attribute serialNumber : String;
        attribute mass : Real;
    }

    part def Transmission specializes Component {
        attribute serialNumber : String;
        attribute mass : Real;
    }

    part def Wheel specializes Component {
        attribute serialNumber : String;
        attribute mass : Real;
    }

    part def Body specializes Component {
        attribute serialNumber : String;
        attribute mass : Real;
    }

    part def Chassis specializes Component {
        attribute serialNumber : String;
        attribute mass : Real;
    }

    part def MassFilter {
        attribute threshold : Real;
    }

    part def Vehicle {
        attribute vin : String;
        attribute ownMass : Real;
        attribute totalMass : Real;
        attribute massThreshold : Real;

        part engine : Engine {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part transmission : Transmission {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part body : Body {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part chassis : Chassis {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part wheel1 : Wheel {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part wheel2 : Wheel {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part wheel3 : Wheel {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part wheel4 : Wheel {
            attribute serialNumber : String;
            attribute mass : Real;
        }

        part massFilter : MassFilter {
            attribute threshold : Real;
        }
    }

    part def VehicleFleet {
        part vehicle1 : Vehicle {
            attribute vin : String = "VIN-001";
            attribute ownMass : Real = 500.0;
            attribute totalMass : Real = 1800.0;
            attribute massThreshold : Real = 50.0;

            part engine : Engine {
                attribute serialNumber : String = "ENG-001";
                attribute mass : Real = 250.0;
            }

            part transmission : Transmission {
                attribute serialNumber : String = "TRN-001";
                attribute mass : Real = 80.0;
            }

            part body : Body {
                attribute serialNumber : String = "BOD-001";
                attribute mass : Real = 400.0;
            }

            part chassis : Chassis {
                attribute serialNumber : String = "CHS-001";
                attribute mass : Real = 300.0;
            }

            part wheel1 : Wheel {
                attribute serialNumber : String = "WHL-001";
                attribute mass : Real = 25.0;
            }

            part wheel2 : Wheel {
                attribute serialNumber : String = "WHL-002";
                attribute mass : Real = 25.0;
            }

            part wheel3 : Wheel {
                attribute serialNumber : String = "WHL-003";
                attribute mass : Real = 25.0;
            }

            part wheel4 : Wheel {
                attribute serialNumber : String = "WHL-004";
                attribute mass : Real = 25.0;
            }

            part massFilter : MassFilter {
                attribute threshold : Real = 50.0;
            }
        }

        part vehicle2 : Vehicle {
            attribute vin : String = "VIN-002";
            attribute ownMass : Real = 600.0;
            attribute totalMass : Real = 2100.0;
            attribute massThreshold : Real = 30.0;

            part engine : Engine {
                attribute serialNumber : String = "ENG-002";
                attribute mass : Real = 300.0;
            }

            part transmission : Transmission {
                attribute serialNumber : String = "TRN-002";
                attribute mass : Real = 100.0;
            }

            part body : Body {
                attribute serialNumber : String = "BOD-002";
                attribute mass : Real = 450.0;
            }

            part chassis : Chassis {
                attribute serialNumber : String = "CHS-002";
                attribute mass : Real = 350.0;
            }

            part wheel1 : Wheel {
                attribute serialNumber : String = "WHL-005";
                attribute mass : Real = 30.0;
            }

            part wheel2 : Wheel {
                attribute serialNumber : String = "WHL-006";
                attribute mass : Real = 30.0;
            }

            part wheel3 : Wheel {
                attribute serialNumber : String = "WHL-007";
                attribute mass : Real = 30.0;
            }

            part wheel4 : Wheel {
                attribute serialNumber : String = "WHL-008";
                attribute mass : Real = 30.0;
            }

            part massFilter : MassFilter {
                attribute threshold : Real = 30.0;
            }
        }
    }
}