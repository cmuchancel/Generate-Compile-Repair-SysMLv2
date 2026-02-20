package VehicleModeling {
  private import ScalarValues::*;

  enum def FuelType {
    enum Gasoline;
    enum Diesel;
    enum Electric;
    enum Hydrogen;
  }

  part def Fuel {
    attribute fuelType : FuelType;
    attribute description : String;
  }

  part def Personnel {
    attribute name : String;
    attribute personnelId : String;
  }

  part def Driver {
    part info : Personnel;
    attribute licenseId : String;
  }

  part def QualityAttributes {
    attribute safetyLevel : String;
    attribute reliabilityLevel : String;
  }

  part def FuelTank {
    part storedFuel : Fuel;
    attribute capacity : Real;
  }

  part def Vehicle {
    attribute model : String;
    part quality : QualityAttributes;
    part driver : Driver;
    part fuelTank : FuelTank;
  }

  part def DataRepository {
    part fuelInfo : Fuel;
    part personnelInfo : Personnel;
  }

  part def VehicleModelingSystem {
    part vehicle : Vehicle;
    part data : DataRepository;
  }
}