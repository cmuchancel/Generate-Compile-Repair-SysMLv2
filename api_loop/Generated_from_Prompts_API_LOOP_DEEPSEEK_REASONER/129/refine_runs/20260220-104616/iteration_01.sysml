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

    part def FrontSeatBelt {
        attribute functionalArea : FunctionalArea = Safety;
        attribute isMandatory : Boolean = true;
    }

    part def Bumper {
        attribute functionalArea : FunctionalArea = Safety;
        attribute isMandatory : Boolean = true;
    }

    part def DriverAirbag {
        attribute functionalArea : FunctionalArea = Safety;
        attribute isMandatory : Boolean = false;
    }

    part def AntiLockBrakingSystem {
        attribute functionalArea : FunctionalArea = Safety;
        attribute isMandatory : Boolean = false;
    }

    part def AlarmSystem {
        attribute functionalArea : FunctionalArea = AntiTheft;
        attribute isMandatory : Boolean = false;
    }

    part def KeylessEntrySystem {
        attribute functionalArea : FunctionalArea = AntiTheft;
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