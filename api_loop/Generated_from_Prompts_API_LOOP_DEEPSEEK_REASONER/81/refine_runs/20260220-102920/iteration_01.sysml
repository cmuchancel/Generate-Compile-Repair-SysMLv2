package AutomotiveConfiguration {
    private import ScalarValues::*;
    
    enum def TransmissionType {
        enum manual;
        enum automatic;
    }
    
    enum def CylinderDiameter {
        enum "70mm";
        enum "100mm";
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
        part engine : Engine;
        part transmission : Transmission;
    }
    
    part def VehicleConfiguration {
        part vehicle : Vehicle {
            part engine : FourCylinderEngine {
                part engine : Engine {
                    part cylinders[4] : Cylinder;
                }
            }
            part transmission : ManualTransmission;
        }
    }
    
    part def VehicleConfiguration2 {
        part vehicle : Vehicle {
            part engine : SixCylinderEngine {
                part engine : Engine {
                    part cylinders[6] : Cylinder;
                }
            }
            part transmission : AutomaticTransmission;
        }
    }
}