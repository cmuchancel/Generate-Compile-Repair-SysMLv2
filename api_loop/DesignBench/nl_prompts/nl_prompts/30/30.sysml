package 'DigitalCameraShooting30' {
	private import ScalarValues::*;

	part def DigitalCameraShootingSystem {
		ref part user : User;
		part sceneSelection : SceneSelection;
		part viewfinderModule : ViewfinderModule;
		part focusingModule : FocusingModule;
		part imagingModule : ImagingModule;
		part photoOutput : PhotoOutputModule;
		attribute singleOperationSufficient : Boolean;
		attribute automatedProcessProvided : Boolean;
		attribute manualAdjustmentNotRequired : Boolean;
		attribute modulesCollaborate : Boolean;
		attribute efficientExperienceDelivered : Boolean;
		attribute convenientExperienceDelivered : Boolean;
		attribute highQualityPhotosProduced : Boolean;
	}

	part def SceneSelection {
		port scenePort : ScenePort;
		attribute sceneSelectedByUser : Boolean;
	}

	part def ViewfinderModule {
		port inputScene : ScenePort;
		port framedScene : ScenePort;
		attribute viewfindingCompleted : Boolean;
	}

	part def FocusingModule {
		port sceneInput : ScenePort;
		port generatedImage : ImagePort;
		attribute requirementsSatisfied : Boolean;
	}

	part def ImagingModule {
		port imageInput : ImagePort;
		port photoOutput : PhotoPort;
		attribute imagingExecuted : Boolean;
	}

	part def PhotoOutputModule {
		port deliveredPhoto : PhotoPort;
	}

	part def User {
		port photoReceipt : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def ImagePort {
		attribute imageData : CameraImage;
	}

	port def PhotoPort {
		attribute photoData : Photograph;
	}

	attribute def SceneDescription;
	attribute def CameraImage;
	attribute def Photograph;

	connection def SelectionToViewfinder {
		end fromSelection : SceneSelection::scenePort;
		end toViewfinder : ViewfinderModule::inputScene;
	}

	connection def ViewfinderToFocusing {
		end fromViewfinder : ViewfinderModule::framedScene;
		end toFocusing : FocusingModule::sceneInput;
	}

	connection def FocusingToImaging {
		end fromFocusing : FocusingModule::generatedImage;
		end toImaging : ImagingModule::imageInput;
	}

	connection def ImagingToOutput {
		end fromImaging : ImagingModule::photoOutput;
		end toOutput : PhotoOutputModule::deliveredPhoto;
	}

	connection def OutputToUser {
		end fromOutput : PhotoOutputModule::deliveredPhoto;
		end toUser : User::photoReceipt;
	}
}
