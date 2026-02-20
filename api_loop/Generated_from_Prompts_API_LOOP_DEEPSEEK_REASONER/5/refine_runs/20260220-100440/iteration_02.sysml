package TrafficSignalSystem {
    private import ScalarValues::*;

    enum def TrafficLightColor {
        enum green;
        enum yellow;
        enum red;
    }

    part def TrafficSignalComponent {
        attribute currentColor : TrafficLightColor;
    }

    part alwaysGreenSignal : TrafficSignalComponent {
        currentColor = .green;
    }
}