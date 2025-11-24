package 'SceneCaptureSystem' {
	item def Scene;
	item def Image;
	item def Photo;
	
	action def FocusScene {
		in scene : Scene;
		out image : Image;
	}
	
	action def CapturePhoto {
		in image : Image;
		out photo : Photo;
	}
	
	part captureWorkflow {
		perform action focusScene : FocusScene;
		perform action capturePhoto : CapturePhoto {
			in image = focusScene.image;
		}
	}
}
