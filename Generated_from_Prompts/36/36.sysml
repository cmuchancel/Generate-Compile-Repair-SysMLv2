package 'VehicleOperationalStateManagement' {
	attribute def VehicleStartSignal;
	attribute def VehicleOnSignal;
	attribute def VehicleOffSignal;
	
	state def VehicleStates;
	
	part vehicleController {
		state vehicleStates : VehicleStates {
			entry action initial;
			transition initial then off;
			state off;
			transition 'off-starting'
				first off
				accept VehicleStartSignal
				then starting;
			state starting;
			transition 'starting-on'
				first starting
				accept VehicleOnSignal
				then on;
			state on;
			transition 'on-off'
				first on
				accept VehicleOffSignal
				then off;
		}
	}
}
