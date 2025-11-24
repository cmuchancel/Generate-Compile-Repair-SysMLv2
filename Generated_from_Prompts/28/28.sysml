package 'BatteryChargingControl' {
	private import ScalarValues::Real;
	
	action def MonitorBatteryLevel {
		out levelPercent : Real;
	}
	
	action def ReplenishBattery;
	action def TerminateCharging;
	
	part batteryCharger {
		attribute batteryLevel : Real;
		action monitorLevel : MonitorBatteryLevel;
		action replenishBattery : ReplenishBattery;
		action terminateCharging : TerminateCharging;
		state chargingProcess {
			entry action initial;
			transition initial then charging;
			state charging {
				do monitorLevel;
			}
			transition 'charging-loop'
				first charging
				accept when batteryCharger.batteryLevel < 100.0
				do replenishBattery
				then charging;
			transition 'charging-complete'
				first charging
				accept when batteryCharger.batteryLevel >= 100.0
				then charged;
			state charged {
				entry terminateCharging;
			}
		}
	}
}
