package 'ConditionalPhotography' {
	private import ScalarValues::Boolean;
	item def Scene;
	item def Image;
	item def Photograph;
	
	action def FocusScene {
		in scene : Scene;
		out image : Image;
		out focusClear : Boolean;
	}
	
	action def ShootPhoto {
		in image : Image;
		out photo : Photograph;
	}
	
	constraint def FocusMustBeClear {
		in focusClear : Boolean;
		focusClear == true
	}
	
	part photographyWorkflow {
		perform action focusScene : FocusScene;
		perform action shootPhoto : ShootPhoto {
			in image = focusScene.image;
		}
		assert constraint focusRequirement : FocusMustBeClear {
			in focusClear = focusScene.focusClear;
		}
	}
}
