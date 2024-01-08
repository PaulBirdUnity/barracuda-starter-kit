using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Menu : MonoBehaviour
{
    string[] sceneNames = new string[]
    {
        "01-StaticImageRecognition-MobileNetV2",
        "02-FaceTracking-BlazeFace",
        "03-MultiObjectStaticImageRecognition-YoloV3Tiny"
    };
    // Start is called before the first frame update
    public void RunDemo(int ID)
    {
        SceneManager.LoadScene(sceneNames[ID]);
    }
}
