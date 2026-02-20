package 'CameraInfoProcessing33' {
	private import ScalarValues::*;

	part def CameraInfoProcessingSystem {
		ref part user : User;
		part viewPort : ViewPort;
		part focusingUnit : FocusingUnit;
		part captureUnit : CaptureUnit;
		part displayPort : DisplayPort;
		attribute automaticProcessingProvided : Boolean;
		attribute technicalDetailsHidden : Boolean;
	}

	part def ViewPort {
		port selectedScene : ScenePort;
	}

	part def FocusingUnit {
		port sceneInput : ScenePort;
		port focusedImage : ImagePort;
		attribute focusingPerformed : Boolean;
	}

	part def CaptureUnit {
		port imageInput : ImagePort;
		port producedPicture : PicturePort;
		attribute capturingPerformed : Boolean;
	}

	part def DisplayPort {
		port displayInput : PicturePort;
		attribute screenDisplayProvided : Boolean;
	}

	part def User {
		port sceneSelection : ScenePort;
		port viewedPicture : PicturePort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def ImagePort {
		attribute imageData : Image;
	}

	port def PicturePort {
		attribute pictureData : Picture;
	}

	attribute def SceneDescription;
	attribute def Image;
	attribute def Picture;

	connection def UserSceneToViewPort {
		end fromUser : User::sceneSelection;
		end toViewPort : ViewPort::selectedScene;
	}

	connection def ViewPortToFocusing {
		end fromViewPort : ViewPort::selectedScene;
		end toFocusing : FocusingUnit::sceneInput;
	}

	connection def FocusingToCapture {
		end fromFocusing : FocusingUnit::focusedImage;
		end toCapture : CaptureUnit::imageInput;
	}

	connection def CaptureToDisplay {
		end fromCapture : CaptureUnit::producedPicture;
		end toDisplay : DisplayPort::displayInput;
	}

	connection def DisplayToUser {
		end fromDisplay : DisplayPort::displayInput;
		end toUser : User::viewedPicture;
	}
}
