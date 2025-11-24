package 'CameraShootAndDisplay' {
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
	
	action def DisplayPhoto {
		in photo : Photo;
		out shownPhoto : Photo;
	}
	
	part cameraWorkflow {
		perform action focusScene : FocusScene;
		perform action capturePhoto : CapturePhoto {
			in image = focusScene.image;
		}
		perform action displayPhoto : DisplayPhoto {
			in photo = capturePhoto.photo;
		}
	}
}
