package 'CameraSystem29' {
	private import ScalarValues::*;

	part def CameraSystem {
		ref part user : User;
		part sceneSelector : SceneSelectionInterface;
		part autofocusSubsystem : AutofocusSubsystem;
		part imagingSubsystem : ImagingSubsystem;
		part photoRecorder : PhotoRecorder;
		attribute highQualityPhotosGenerated : Boolean;
		attribute internalDetailsHidden : Boolean;
	}

	part def SceneSelectionInterface {
		port scenePort : ScenePort;
		attribute sceneConfirmedByUser : Boolean;
	}

	part def AutofocusSubsystem {
		port inputScene : ScenePort;
		port clearImage : ImagePort;
		attribute clearRealisticImageObtained : Boolean;
	}

	part def ImagingSubsystem {
		port imageInput : ImagePort;
		port capturedPhoto : PhotoPort;
		attribute imageCapturedAsPhotograph : Boolean;
	}

	part def PhotoRecorder {
		port photoPort : PhotoPort;
		attribute photosRecorded : Boolean;
		attribute photosSaved : Boolean;
	}

	part def User {
		port receivedPhoto : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def ImagePort {
		attribute imageData : ClearImage;
	}

	port def PhotoPort {
		attribute photoData : Photograph;
	}

	attribute def SceneDescription;
	attribute def ClearImage;
	attribute def Photograph;

	connection def SceneToAutofocus {
		end fromSelector : SceneSelectionInterface::scenePort;
		end toAutofocus : AutofocusSubsystem::inputScene;
	}

	connection def AutofocusToImaging {
		end clearImage : AutofocusSubsystem::clearImage;
		end imagingInput : ImagingSubsystem::imageInput;
	}

	connection def ImagingToRecorder {
		end fromImaging : ImagingSubsystem::capturedPhoto;
		end toRecorder : PhotoRecorder::photoPort;
	}

	connection def RecorderToUser {
		end fromRecorder : PhotoRecorder::photoPort;
		end toUser : User::receivedPhoto;
	}
}
