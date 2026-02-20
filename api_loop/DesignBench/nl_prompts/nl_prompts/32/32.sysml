package 'CameraShootingDisplay32' {
	private import ScalarValues::*;

	part def CameraShootingDisplaySystem {
		ref part user : User;
		part sceneInput : SceneInputInterface;
		part focusingProcess : FocusingProcess;
		part captureProcess : CaptureProcess;
		part displayModule : DisplayModule;
		attribute seamlessIntegrationProvided : Boolean;
		attribute automaticOperationProvided : Boolean;
		attribute simpleShootingExperience : Boolean;
	}

	part def SceneInputInterface {
		port scenePort : ScenePort;
		attribute currentSceneProvided : Boolean;
	}

	part def FocusingProcess {
		port inputScene : ScenePort;
		port outboundImage : ImagePort;
		attribute focusingPerformed : Boolean;
	}

	part def CaptureProcess {
		port imageInput : ImagePort;
		port photoOutput : PhotoPort;
		attribute photoGenerated : Boolean;
	}

	part def DisplayModule {
		port displayedPhoto : PhotoPort;
		attribute screenDisplayProvided : Boolean;
	}

	part def User {
		port viewedPhoto : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def ImagePort {
		attribute imageData : FocusedImage;
	}

	port def PhotoPort {
		attribute photoData : Photograph;
	}

	attribute def SceneDescription;
	attribute def FocusedImage;
	attribute def Photograph;

	connection def SceneToFocus {
		end fromSceneInput : SceneInputInterface::scenePort;
		end toFocusing : FocusingProcess::inputScene;
	}

	connection def FocusToCapture {
		end fromFocusing : FocusingProcess::outboundImage;
		end toCapture : CaptureProcess::imageInput;
	}

	connection def CaptureToDisplay {
		end fromCapture : CaptureProcess::photoOutput;
		end toDisplay : DisplayModule::displayedPhoto;
	}

	connection def DisplayToUser {
		end fromDisplay : DisplayModule::displayedPhoto;
		end toUser : User::viewedPhoto;
	}
}
