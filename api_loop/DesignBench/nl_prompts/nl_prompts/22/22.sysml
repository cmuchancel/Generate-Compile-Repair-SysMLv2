package 'PhotographyWorkflow22' {
	private import ScalarValues::*;

	part def PhotographyWorkflowSoftware {
		ref part user : User;
		part sceneInput : SceneInputInterface;
		part focusFunction : FocusFunction;
		part shootFunction : ShootFunction;
		part photoOutput : PhotoOutputInterface;
		attribute compositionSupported : Boolean;
		attribute imagingSupported : Boolean;
		attribute photoOutputSupported : Boolean;
		attribute workflowAutomated : Boolean;
		attribute workflowStreamlined : Boolean;
	}

	part def SceneInputInterface {
		port scenePort : ScenePort;
		attribute sceneProvidedByUser : Boolean;
	}

	part def FocusFunction {
		port capturedScene : ScenePort;
		port generatedImage : ImagePort;
		attribute focusOperationPerformed : Boolean;
	}

	part def ShootFunction {
		port focusedImage : ImagePort;
		port producedPicture : PicturePort;
		attribute shootingOperationPerformed : Boolean;
	}

	part def PhotoOutputInterface {
		port deliveredPicture : PicturePort;
	}

	part def User {
		port receivesPicture : PicturePort;
	}

	port def ScenePort {
		attribute sceneData : Scene;
	}

	port def ImagePort {
		attribute imageData : Image;
	}

	port def PicturePort {
		attribute pictureData : Picture;
	}

	attribute def Scene;
	attribute def Image;
	attribute def Picture;

	connection def SceneToFocus {
		end fromInput : SceneInputInterface::scenePort;
		end toFocus : FocusFunction::capturedScene;
	}

	connection def FocusToShoot {
		end fromFocus : FocusFunction::generatedImage;
		end toShoot : ShootFunction::focusedImage;
	}

	connection def ShootToOutput {
		end fromShoot : ShootFunction::producedPicture;
		end toOutput : PhotoOutputInterface::deliveredPicture;
	}

	connection def OutputToUser {
		end fromOutput : PhotoOutputInterface::deliveredPicture;
		end toUser : User::receivesPicture;
	}
}
