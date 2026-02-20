package TrafficSignalSystem {
    private import ScalarValues::*;

    enum def TrafficColor {
        enum green;
        enum yellow;
        enum red;
    }

    part def TrafficSignal {
        attribute color : TrafficColor;
    }

    part trafficSignalGreen : TrafficSignal {
        attribute redefines color = TrafficColor::green;
    }
}