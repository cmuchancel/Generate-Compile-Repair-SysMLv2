package TrafficSignalSystem {
    private import ScalarValues::*;

    enum def TrafficLightColor {
        enum green;
        enum yellow;
        enum red;
    }

    part def TrafficSignal {
        attribute currentColor : TrafficLightColor;
    }

    part def AlwaysGreenSignal {
        attribute currentColor : TrafficLightColor = TrafficLightColor::green;
    }

    part trafficSignal : TrafficSignal;
    part alwaysGreenSignal : AlwaysGreenSignal;
}