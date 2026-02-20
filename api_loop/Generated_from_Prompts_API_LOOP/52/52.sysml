package VehicleComposite {
  private import ScalarValues::*;

  part def Engine {
  }

  part def Driver {
  }

  part def Vehicle {
    attribute mass : Real;
    attribute gear : Integer;
    attribute accelerator : Real;
  }

  part def VehicleTimeline {
    part initial : Vehicle;
    part next : Vehicle;
  }

  part def CompositeSystem {
    part vehicle_1 : VehicleTimeline;
    part engine_1 : Engine;
    part driver_1 : Driver;
  }

  part SystemInstance : CompositeSystem {
    part vehicle_1 : VehicleTimeline {
      part initial : Vehicle {
        attribute mass = 2000.0;
        attribute gear = 0;
        attribute accelerator = 0.0;
      }
      part next : Vehicle {
        attribute mass = 1500.0;
        attribute gear = 2;
        attribute accelerator = 0.5;
      }
    }
  }
}