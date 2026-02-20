package VehicleComposite
import ScalarValues::*;

part def Engine;
part def Driver;

part def VehicleState {
  attribute gear : Integer;
  attribute accelerator : Real;
};

part def Vehicle {
  attribute mass : Real;
  part state : VehicleState;
};

part def VehicleTimeline {
  part initial : Vehicle;
  part next : Vehicle;
};

part def CompositeSystem {
  part vehicle_1 : VehicleTimeline;
  part engine_1 : Engine;
  part driver_1 : Driver;
};

part SystemInstance : CompositeSystem {
  part vehicle_1 : VehicleTimeline {
    part initial : Vehicle {
      attribute mass = 2000.0;
      part state : VehicleState {
        attribute gear = 0;
        attribute accelerator = 0.0;
      };
    };
    part next : Vehicle {
      attribute mass = 1500.0;
      part state : VehicleState {
        attribute gear = 2;
        attribute accelerator = 0.5;
      };
    };
  };
};

end package