package 'PhotographicWorkflow' {
	item def Scene;
	item def Image;
	item def Photo;
	
	action def FocusingProcess {
		in scene : Scene;
		out image : Image;
	}
	
	action def ShootingProcess {
		in image : Image;
		out photo : Photo;
	}
	
	part workflow {
		perform action focusing : FocusingProcess;
		perform action shooting : ShootingProcess {
			in image = focusing.image;
		}
	}
}
