package VehicleSafetyAndAntiTheftFunctionCatalogue {
    private import ScalarValues::*;

    enum def FunctionalArea {
        enum Safety;
        enum AntiTheft;
        enum Both;
    }

    part def Component {
        attribute functionalArea : FunctionalArea;
        attribute isMandatory : Boolean;
    }

    part def FrontSeatBelt : Component {
        attribute functionalArea : FunctionalArea = FunctionalArea::Safety;
        attribute isMandatory : Boolean = true;
    }

    part def Bumper : Component {
        attribute functionalArea : FunctionalArea = FunctionalArea::Safety;
        attribute isMandatory : Boolean = true;
    }

    part def DriverAirbag : Component {
        attribute functionalArea : FunctionalArea = FunctionalArea::Safety;
        attribute isMandatory : Boolean = false;
    }

    part def AntiLockBrakingSystem : Component {
        attribute functionalArea : FunctionalArea = FunctionalArea::Safety;
        attribute isMandatory : Boolean = false;
    }

    part def AlarmSystem : Component {
        attribute functionalArea : FunctionalArea = FunctionalArea::AntiTheft;
        attribute isMandatory : Boolean = false;
    }

    part def KeylessEntrySystem : Component {
        attribute functionalArea : FunctionalArea = FunctionalArea::AntiTheft;
        attribute isMandatory : Boolean = false;
    }

    part def Vehicle {
        part frontSeatBelt : FrontSeatBelt[2];
        part bumper : Bumper[2];
        part driverAirbag : DriverAirbag[0..1];
        part antiLockBrakingSystem : AntiLockBrakingSystem[0..1];
        part alarmSystem : AlarmSystem[0..1];
        part keylessEntrySystem : KeylessEntrySystem[0..1];
    }
}