package 'TrafficSignalSystem' {
	private import ScalarValues::*;

	part def TrafficSignal {
		attribute color : SignalColorKind;
	}

	part def GreenTrafficSignal specializes TrafficSignal {
		attribute color redefines TrafficSignal::color : SignalColorKind = SignalColorKind::green;
	}

	enum def SignalColorKind {
		enum green;
		enum yellow;
		enum red;
	}
}
