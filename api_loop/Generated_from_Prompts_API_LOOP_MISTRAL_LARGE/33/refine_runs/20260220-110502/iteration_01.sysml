package CameraSystem {
    private import ScalarValues::*;

    part def Camera;
    part def Viewfinder;
    part def Display;
    part def ImageProcessor;

    item def Image;
    item def Picture;

    port def ViewPort;
    port def DisplayPort;

    part camera : Camera {
        port viewPort : ViewPort;
        port displayPort : DisplayPort;
        part viewfinder : Viewfinder;
        part display : Display;
        part processor : ImageProcessor;

        attribute currentImage : Image;
        attribute currentPicture : Picture;
    }

    connect camera.viewPort to camera.viewfinder;
    connect camera.processor to camera.displayPort;
    connect camera.displayPort to camera.display;
}