package AutomotiveConfiguration {
    private import ScalarValues::*;

    enum def TransmissionType {
        enum manual;
        enum automatic;
    }

    enum def CylinderDiameter {
        enum seventy_mm;
        enum one_hundred_mm;
    }

    part def Cylinder {
        attribute diameter : CylinderDiameter;
    }

    part def Engine {
        part cylinders[0..*] : Cylinder;
    }

    part def FourCylinderEngine {
        part engine : Engine {
            part cylinders[4] : Cylinder;
        }
    }

    part def SixCylinderEngine {
        part engine : Engine {
            part cylinders[6] : Cylinder;
        }
    }

    part def Transmission {
        attribute type : TransmissionType;
    }

    part def ManualTransmission {
        attribute type : TransmissionType = TransmissionType::manual;
    }

    part def AutomaticTransmission {
        attribute type : TransmissionType = TransmissionType::automatic;
    }

    part def Vehicle {
        part powertrain : Engine;
        part gearbox : Transmission;
    }

    part def ValidConfiguration1 {
        part car : Vehicle {
            part powertrain : FourCylinderEngine {
                part engine : Engine {
                    part cylinders[4] : Cylinder;
                }
            }
            part gearbox : ManualTransmission;
        }
    }

    part def ValidConfiguration2 {
        part car : Vehicle {
            part powertrain : SixCylinderEngine {
                part engine : Engine {
                    part cylinders[6] : Cylinder;
                }
            }
            part gearbox : AutomaticTransmission;
        }
    }
}