package 'CameraAutofocusImaging' {
	item def Scene;
	item def Image;
	item def Photograph;
	
	action def Autofocus {
		in scene : Scene;
		out image : Image;
	}
	
	action def CaptureImage {
		in image : Image;
		out photo : Photograph;
	}
	
	action def RecordPhoto {
		in photo : Photograph;
		out savedPhoto : Photograph;
	}
	
	part camera {
		part autofocusSubsystem {
			perform action focusScene : Autofocus;
		}
		part imagingSubsystem {
			perform action capturePhoto : CaptureImage {
				in image = autofocusSubsystem.focusScene.image;
			}
		}
		part photoArchive {
			item savedPhotos : Photograph[*] ordered;
		}
		perform action recordPhoto : RecordPhoto {
			in photo = imagingSubsystem.capturePhoto.photo;
			out savedPhoto = photoArchive.savedPhotos;
		}
	}
}
