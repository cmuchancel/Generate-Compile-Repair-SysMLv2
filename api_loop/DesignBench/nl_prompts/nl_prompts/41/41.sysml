package 'TimedServer41' {
	private import ScalarValues::*;

	part def TimedServerSystem {
		part server : TimedServer;
		part schedule : DailySchedule;
		part requestInterface : RequestPort;
		attribute responseTimingControllable : Boolean;
		attribute orderlyOperationMaintained : Boolean;
	}

	part def TimedServer {
		part localClock : LocalClock;
		port commandPort : RequestPort;
		attribute offState : Boolean;
		attribute waitingState : Boolean;
		attribute responseState : Boolean;
		attribute startRequiredForWaiting : Boolean;
		attribute waitDurationFiveMinutes : Boolean;
	}

	part def LocalClock {
		attribute currentTime : TimeOfDay;
	}

	part def DailySchedule {
		attribute revertTime : TimeOfDay;
		attribute automaticReversionEnabled : Boolean;
		attribute multipleCyclesSupported : Boolean;
	}

	port def RequestPort {
		attribute startCommandReceived : Boolean;
		attribute requestMessageReceived : Boolean;
		attribute shutdownTimeCommand : Boolean;
	}

	attribute def TimeOfDay {
		attribute hour : Integer;
		attribute minute : Integer;
		attribute second : Integer;
	}
}
