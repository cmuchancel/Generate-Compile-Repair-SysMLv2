package 'ServerTimedBehavior' {
	private import ScalarValues::Boolean;
	private import ScalarValues::Real;
	private import Time::*;
	
	port def RequestPort {
		attribute startCommand : Boolean;
		attribute requestCommand : Boolean;
	}
	
	part server {
		attribute localClock : Time::DateTime;
		attribute shutdownTime : Time::DateTime;
		attribute responseElapsedMinutes : Real;
		attribute responseDurationMinutes : Real = 5.0;
		port requestPort : RequestPort;
		
		state serverStates {
			entry action initial;
			transition initial then off;
			state off;
			transition 'off-waiting'
				first off
				accept when requestPort.startCommand
				then waiting;
			state waiting;
			transition 'waiting-response'
				first waiting
				accept when requestPort.requestCommand
				then response;
			transition 'waiting-off'
				first waiting
				accept at shutdownTime
				then off;
			state response;
			transition 'response-waiting'
				first response
				accept when responseElapsedMinutes >= responseDurationMinutes
				then waiting;
			transition 'response-off'
				first response
				accept at shutdownTime
				then off;
		}
	}
}
