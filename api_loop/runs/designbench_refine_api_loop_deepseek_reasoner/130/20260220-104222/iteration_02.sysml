package VehicleSystem {
    private import ScalarValues::*;

    enum def Classification {
        enum Safety;
        enum Security;
    }

    part def SeatBelt {
        attribute classification : Classification = Classification::Safety;
        attribute mandatory : Boolean = true;
    }

    part def Airbag {
        attribute classification : Classification = Classification::Safety;
        attribute mandatory : Boolean = false;
    }

    part def AntiTheftAlarm {
        attribute classification : Classification = Classification::Security;
        attribute mandatory : Boolean = true;
    }

    part def Bumper {
        attribute classification : Classification = Classification::Safety;
        attribute mandatory : Boolean = true;
    }

    part def KeylessEntrySystem {
        attribute classification : Classification = Classification::Security;
        attribute mandatory : Boolean = true;
    }

    part def Wheel {
        attribute classification : Classification = Classification::Safety;
        attribute mandatory : Boolean = true;
    }

    part def AntiLockBrakingSystem {
        attribute classification : Classification = Classification::Safety;
        attribute mandatory : Boolean = false;
    }

    part def Vehicle {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part airbag : Airbag;
        part antiTheftAlarm : AntiTheftAlarm;
        part bumper : Bumper;
        part keylessEntrySystem : KeylessEntrySystem;
        part wheel1 : Wheel;
        part wheel2 : Wheel;
        part abs1 : AntiLockBrakingSystem;
        part abs2 : AntiLockBrakingSystem;
    }
}