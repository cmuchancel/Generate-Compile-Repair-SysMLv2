package SensorSystem
{
  private import ScalarValues::*;

  enum def SensorStatus
  {
    enum Ready;
    enum NotReady;
    enum Updating;
  }

  part def Sensor
  {
    attribute status : SensorStatus = SensorStatus::NotReady;
    attribute updated : Boolean = false;
  }

  part def SensorManager
  {
    part sensor1 : Sensor;
    part sensor2 : Sensor;
    part sensor3 : Sensor;
    attribute autoUpdateEnabled : Boolean = true;
    attribute autoTraverseEnabled : Boolean = true;
  }

  part system : SensorManager;
}