package 'PhotographyWorkflowSystem' {
	item def Scene;
	item def Image;
	item def Picture;
	
	action def Focus {
		in scene : Scene;
		out image : Image;
	}
	
	action def Shoot {
		in image : Image;
		out picture : Picture;
	}
	
	part photographyWorkflow {
		perform action focus : Focus;
		perform action shoot : Shoot {
			in image = focus.image;
		}
	}
}
