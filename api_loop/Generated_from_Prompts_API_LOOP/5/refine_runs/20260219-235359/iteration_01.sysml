package TrafficSignalSystemPkg {
    private import ScalarValues::*;
    enum def TrafficColor {
        enum green;
        enum yellow;
        enum red;
    }
    part def TrafficSignal {
        attribute color : TrafficColor;
    }
    part def TrafficSignalSystem {
        part signal : TrafficSignal;
        part greenSignal : TrafficSignal {
            attribute color : TrafficColor = TrafficColor::green;
        }
    }
}