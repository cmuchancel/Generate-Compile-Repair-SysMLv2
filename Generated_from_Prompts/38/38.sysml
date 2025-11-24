package 'VehicleStateControl' {
	attribute def VehicleStartSignal;
	attribute def VehicleStartedSignal;
	attribute def VehicleShutdownSignal;
	
	state def VehicleStateMachine;
	
	part vehicleController {
		state vehicleState : VehicleStateMachine {
			entry action initial;
			transition initial then shutdown;
			state shutdown;
			transition 'shutdown-startup'
				first shutdown
				accept VehicleStartSignal
				then startup;
			state startup;
			transition 'startup-running'
				first startup
				accept VehicleStartedSignal
				then running;
			state running;
			transition 'running-shutdown'
				first running
				accept VehicleShutdownSignal
				then shutdown;
		}
	}
}
