package 'SensorStatusManagement' {
	private import ScalarValues::Boolean;
	private import ControlFunctions::forAll;
	
	part def Sensor {
		attribute isReady : Boolean;
		attribute updated : Boolean;
	}
	
	action def MonitorSensors {
		in sensors : Sensor[*];
		out readySensors : Sensor[*];
	}
	
	action def UpdateSensors {
		in sensors : Sensor[*];
	}
	
	constraint def ReadySensorsUpdated {
		in sensors : Sensor[*];
		sensors->forAll { in ref s : Sensor; s.isReady implies s.updated }
	}
	
	part sensorManager {
		part sensors : Sensor[*];
		perform action monitorSensors : MonitorSensors {
			in sensors = sensorManager.sensors;
		}
		perform action updateSensors : UpdateSensors {
			in sensors = sensorManager.sensors;
		}
		constraint readySensorsRule : ReadySensorsUpdated {
			in sensors = sensorManager.sensors;
		}
	}
}
