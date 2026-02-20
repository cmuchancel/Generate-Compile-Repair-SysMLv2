package TrafficSignalSystem {
    private import ScalarValues::*;

    enum def TrafficLightColor {
        green,
        yellow,
        red
    }

    part def TrafficSignalComponent {
        attribute currentColor : TrafficLightColor;
    }

    part alwaysGreenSignal : TrafficSignalComponent {
        currentColor = TrafficLightColor::green;
    }
}