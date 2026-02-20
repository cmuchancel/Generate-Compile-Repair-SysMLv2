package 'SceneCaptureSystem18' {
	private import ScalarValues::*;

	part def SceneCaptureSystem {
		ref part user : User;
		part sceneSelection : SceneSelectionInterface;
		part focusingModule : AutoFocusingModule;
		part imagingModule : PhotoCaptureModule;
		part photoOutput : PhotoOutputChannel;
		attribute automaticFocusing : Boolean;
		attribute automaticShooting : Boolean;
		attribute manualInterventionEliminated : Boolean;
	}

	part def SceneSelectionInterface {
		port selectedScenePort : ScenePort;
	}

	part def AutoFocusingModule {
		port inputScenePort : ScenePort;
		port focusedImagePort : FocusedImagePort;
		attribute intermediateImageGenerated : Boolean;
	}

	part def PhotoCaptureModule {
		port inputImagePort : FocusedImagePort;
		port finalPhotoPort : PhotoPort;
	}

	part def PhotoOutputChannel {
		port deliveredPhotoPort : PhotoPort;
	}

	part def User {
		port receivedPhotoPort : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def FocusedImagePort {
		attribute focusedImage : FocusedImage;
	}

	port def PhotoPort {
		attribute photoData : CapturedPhoto;
	}

	attribute def SceneDescription;
	attribute def FocusedImage;
	attribute def CapturedPhoto;

	connection def SceneSelectionToFocusing {
		end fromSelection : SceneSelectionInterface::selectedScenePort;
		end toFocusing : AutoFocusingModule::inputScenePort;
	}

	connection def FocusingToImaging {
		end fromFocusing : AutoFocusingModule::focusedImagePort;
		end toImaging : PhotoCaptureModule::inputImagePort;
	}

	connection def ImagingToOutput {
		end fromImaging : PhotoCaptureModule::finalPhotoPort;
		end toOutput : PhotoOutputChannel::deliveredPhotoPort;
	}

	connection def OutputToUser {
		end fromOutput : PhotoOutputChannel::deliveredPhotoPort;
		end toUser : User::receivedPhotoPort;
	}
}
