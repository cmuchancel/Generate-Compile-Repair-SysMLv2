package 'AutomatedPhotoCapture' {
	item def Scene;
	item def Image;
	item def Photograph;
	
	action def AcquireScene {
		out scene : Scene;
	}
	
	action def FocusScene {
		in scene : Scene;
		out image : Image;
	}
	
	action def ShootPhoto {
		in image : Image;
		out photo : Photograph;
	}
	
	action def DisplayPhoto {
		in photo : Photograph;
		out displayedPhoto : Photograph;
	}
	
	part photoWorkflow {
		perform action acquireScene : AcquireScene;
		perform action focusScene : FocusScene {
			in scene = acquireScene.scene;
		}
		perform action shootPhoto : ShootPhoto {
			in image = focusScene.image;
		}
		perform action displayPhoto : DisplayPhoto {
			in photo = shootPhoto.photo;
		}
	}
}
