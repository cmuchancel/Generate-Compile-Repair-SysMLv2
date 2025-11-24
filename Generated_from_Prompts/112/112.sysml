package 'VehiclePowerControl' {
	private import ScalarValues::*;
	private import ISQ::*;
	package Definitions {
		part def VehicleA {
			perform action 'perform self check': 'Perform Self Check';
			perform action 'provide power': 'Provide Power';
			perform action 'apply parking brake': 'Apply Parking Brake';
			perform action 'sense temperature': 'Sense Temperature';
			exhibit state 'vehicle states': 'Vehicle States';
		}
		part def VehicleController {
			exhibit state 'controller states': 'Controller States';
		}
		state def 'Vehicle States';
		state def 'Controller States';
		action def 'Perform Self Check';
		action def 'Provide Power';
		action def 'Apply Parking Brake';
		action def 'Sense Temperature' { out temperature : TemperatureValue; }
		action def 'Notify Controller Start' { in controller : VehicleController; }
		action def 'Notify Controller Off' { in controller : VehicleController; }
		action def 'Notify Controller Over Temperature' { in controller : VehicleController; }
		attribute def FuelCommand;
		occurrence def 'Vehicle Start Signal';
		occurrence def 'Vehicle On Signal';
		occurrence def 'Vehicle Off Signal';
		occurrence def 'Start Signal';
		occurrence def 'Off Signal';
		occurrence def 'Maintenance Signal';
		occurrence def 'Restore to Normal';
		occurrence def 'Over Temperature Signal';
	}
	package Usages {
		private import Definitions::*;
		action 'perform self check': 'Perform Self Check';
		action 'provide power': 'Provide Power';
		action 'apply parking brake': 'Apply Parking Brake';
		action 'sense temperature': 'Sense Temperature';
		action 'notify controller start': 'Notify Controller Start';
		action 'notify controller off': 'Notify Controller Off';
		action 'notify controller over temperature': 'Notify Controller Over Temperature';
		state 'vehicle states': 'Vehicle States' parallel {
			state 'operational states' {
				entry action initial;
				transition initial then off;
				state off;
				transition 'off-starting'
					first off
					accept 'Vehicle Start Signal'
					if vehicleA_c1.'brake pedal pressed'
					do 'notify controller start' { in controller = vehicleA_c1.vehicleController; }
					then starting;
				state starting;
				transition 'starting-running'
					first starting
					accept 'Vehicle On Signal'
					then running;
				state running {
					entry 'perform self check';
					do 'provide power';
					exit 'apply parking brake';
				}
				transition 'running-off'
					first running
					accept 'Vehicle Off Signal'
					do 'notify controller off' { in controller = vehicleA_c1.vehicleController; }
					then off;
			}
			state 'health states' {
				entry action initial;
				transition initial then normal;
				state normal;
				transition 'normal-maintenance'
					first normal
					accept 'Maintenance Signal'
					then maintenance;
				transition 'normal-degraded'
					first normal
					accept when 'sense temperature'.temperature > vehicleA_c1.Tmax
					do 'notify controller over temperature' { in controller = vehicleA_c1.vehicleController; }
					then degraded;
				state maintenance;
				transition 'maintenance-normal'
					first maintenance
					accept 'Restore to Normal'
					then normal;
				state degraded;
				transition 'degraded-normal'
					first degraded
					accept 'Restore to Normal'
					then normal;
				do 'sense temperature' { out temperature; }
			}
		}
		state 'controller states': 'Controller States' {
			entry action initial;
			transition initial then off;
			state off;
			transition 'off-on'
				first off
				accept 'Start Signal'
				then on;
			state on;
			transition 'on-off'
				first on
				accept 'Off Signal'
				then off;
		}
		part vehicleA_c1 : VehicleA {
			attribute 'brake pedal pressed' : Boolean;
			attribute Tmax : TemperatureValue;
			part vehicleController : VehicleController {
				exhibit 'controller states' :>> VehicleController::'controller states';
			}
			exhibit 'vehicle states' :>> VehicleA::'vehicle states';
		}
	}
}
