package 'VehicleStartStopManagement' {
	attribute def 'Vehicle Start Signal';
	attribute def 'Vehicle Power On Signal';
	attribute def 'Vehicle Shutdown Signal';
	
	state def 'Vehicle States';
	
	part vehicleController {
		state 'vehicle states' : 'Vehicle States' {
			entry action initial;
			transition initial then off;
			state off;
			transition 'off-starting'
				first off
				accept 'Vehicle Start Signal'
				then starting;
			state starting;
			transition 'starting-on'
				first starting
				accept 'Vehicle Power On Signal'
				then on;
			state on;
			transition 'starting-off'
				first starting
				accept 'Vehicle Shutdown Signal'
				then off;
			transition 'on-off'
				first on
				accept 'Vehicle Shutdown Signal'
				then off;
		}
	}
}
