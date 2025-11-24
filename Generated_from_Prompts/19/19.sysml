package 'ScenePhotographyWorkflow' {
	item def Scene;
	item def Image;
	item def Photograph;
	
	action def Focusing {
		in scene : Scene;
		out image : Image;
	}
	
	action def Shooting {
		in image : Image;
		out photo : Photograph;
	}
	
	part workflow {
		perform action focusing : Focusing;
		perform action shooting : Shooting {
			in image = focusing.image;
		}
	}
}
