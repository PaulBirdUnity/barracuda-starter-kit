﻿using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using Object = System.Object;
using UnityEngine.SceneManagement;
using System.Threading.Tasks;
using UnityEngine.Video;

public class RunInferenceYOLO : MonoBehaviour
{
    public Texture2D[] inputImage;
    public RawImage displayImage;
    public ModelAsset srcModel;
    public TextAsset labelsAsset;
    public Dropdown backendDropdown;
    public Transform displayLocation;
    public Font font;
    public float confidenceThreshold = 0.25f;
    public float iouThreshold = 0.45f;
    public Sprite boxTexture;
    public Text FPStext;

    enum InputType { Image, Video, Webcam };
    InputType inputType = InputType.Video;

    private int selectedImage = -1;
    private Model model;
    private IWorker engine;
    private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
    private string[] labels;
    private RenderTexture targetRT;
    private string inferenceBackend = "GPUCompute";
    private const int amountOfClasses = 80;
    private const int box20Sections = 20;
    private const int box40Sections = 40;
    private const int anchorBatchSize = 85;
    private const int inputResolutionX = 640;
    private const int inputResolutionY = 640;
    //model output returns box scales relative to the anchor boxes, 3 are used for 40x40 outputs and other 3 for 20x20 outputs,
    //each cell has 3 boxes 3x85=255
    private readonly float[] anchors = {10,14, 23,27, 37,58, 81,82, 135,169, 344,319};
    private VideoPlayer video;
    //box struct with the original output data
    public struct Box
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public string label;
        public int anchorIndex;
        public int cellIndexX;
        public int cellIndexY;
    }
    
    //restructured data with pixel units
    public struct PixelBox
    {
        public float x;
        public float y;
        public float width;
        public float height;
        public string label;
    }
    
    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;
        
        //parse neural net labels
        labels = labelsAsset.text.Split('\n');
        //load model
        model = ModelLoader.Load(srcModel);

        targetRT = new RenderTexture(640, 640, 0);
        AddBackendOptions();
        SelectBackend();

        SetupInput();
        ExecuteML(0);
    }
    void SetupInput()
    {
        video = gameObject.AddComponent<VideoPlayer>();
        video.renderMode = VideoRenderMode.APIOnly;
        video.source = VideoSource.Url;
        video.url = Application.streamingAssetsPath + "/giraffes.mp4";
        video.isLooping = true;
        video.Play();
    }

    System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();

    void SetupEngine()
    {
        engine?.Dispose();
        if (inferenceBackend == "CPU")
        {
            engine = WorkerFactory.CreateWorker(BackendType.CPU, model);
        }
        else if (inferenceBackend == "GPUCompute")
        {
            engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        }
        else if (inferenceBackend == "PixelShader")
        {
            engine = WorkerFactory.CreateWorker(BackendType.GPUPixel, model);
        }
    }

    private void Update()
    {
        N = 0;
        if(inputType == InputType.Video)
        {
            ExecuteML(0);
            FPStext.text = Mathf.FloorToInt(FPS.GetCurrentFPS()+0.5f) + " FPS";
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            CleanUp();
            SceneManager.LoadScene("Menu");
            return;
        }
    }


    public void ExecuteML(int imageID)
    {
        if (inputType == InputType.Image && imageID == selectedImage) return; //could cause a bug if toggled on sent first
        watch.Reset(); watch.Start();
        ClearAnnotations();
        selectedImage = imageID;
        if (inputImage[selectedImage].width != 640 || inputImage[selectedImage].height != 640)
        {
            Debug.LogError("Image resolution must be 640x640. Make sure Texture Import Settings are similar to the example images");
        }
        displayImage.texture = inputImage[selectedImage];

        Texture texture = inputImage[imageID];
        if (video && video.texture)
        {
            float aspect = video.width*1f / video.height;
            Graphics.Blit(video.texture, targetRT, new Vector2(1f/aspect, 1), new Vector2(0, 0));
            texture = targetRT;// video.texture;
            displayImage.texture = texture;
        }
        else
        {
            return;
        }


        //preprocess image for input
        using var input = TextureConverter.ToTensor(texture, 640, 640, 3);
        engine.Execute(input);

        //read output tensors

        //Output tensor name for 20x20 boundingBoxes:
        var output20 = engine.PeekOutput("016_convolutional") as TensorFloat;

        //Output tensor name for 40x40 boundingBoxes
        var output40 = engine.PeekOutput("023_convolutional") as TensorFloat;

        //this list is used to store the original model output data
        List<Box> outputBoxList = new List<Box>();
        
        //this list is used to store the values converted to intuitive pixel data
        List<PixelBox> pixelBoxList = new List<PixelBox>();

        //decode the output 
        outputBoxList = DecodeOutput(output20, output40);
        
        //convert output to intuitive pixel data (x,y coords from the center of the image; height and width in pixels)
        pixelBoxList = ConvertBoxToPixelData(outputBoxList);
        
        //non max suppression (remove overlapping objects)
        pixelBoxList = NonMaxSuppression(pixelBoxList);

        //draw bounding boxes
        for (int i = 0; i < pixelBoxList.Count; i++)
        {
            DrawBox(pixelBoxList[i]);
        }

        

        //clean memory
        input.Dispose();
        //Resources.UnloadUnusedAssets();

        FPStext.text = $"Time taken {watch.ElapsedMilliseconds} ms";
        //Debug.Log("DoneML=" + watch.ElapsedMilliseconds / 1000f);
    }

    public void AddBackendOptions()
    {
        List<string> options = new List<string> ();
        options.Add("CPU");
        #if !UNITY_WEBGL
        options.Add("GPUCompute");
        #endif
        options.Add("PixelShader");
        backendDropdown.ClearOptions ();
        backendDropdown.AddOptions(options);
        backendDropdown.value = 1;
    }
    
    public void SelectBackend()
    {
        if (backendDropdown.options[backendDropdown.value].text == "CPU")
        {
            inferenceBackend = "CPU";
        }
        else if (backendDropdown.options[backendDropdown.value].text == "GPUCompute")
        {
            inferenceBackend = "GPUCompute";
        }
        else if (backendDropdown.options[backendDropdown.value].text == "PixelShader")
        {
            inferenceBackend = "PixelShader";
        }
        
        SetupEngine();
    }

    public List<Box> DecodeOutput(TensorFloat output20, TensorFloat output40)
    {
        List<Box> outputBoxList = new List<Box>();
        
        //decode results into a list for each output(20x20 and 40x40), anchor mask selects the output box presets (first 3 or the last 3 presets) 
        outputBoxList = DecodeYolo(outputBoxList, output40, box40Sections, 0);
        outputBoxList = DecodeYolo(outputBoxList, output20, box20Sections, 3);
        
        return outputBoxList;
    }

    public List<Box> DecodeYolo(List<Box> outputBoxList, TensorFloat output, int boxSections, int anchorMask )
    {
        output.MakeReadable();

        for (int anchor = 0; anchor < 3; anchor++)
        {
            int N = anchor * anchorBatchSize;
            for (int x = 0; x < boxSections; x++)
            {
                for (int y = 0; y < boxSections; y++)
                {
                    if (output[0, N + 4, x, y] > confidenceThreshold)
                    {
                        //Identify the best class
                        //GetMaxIndex( output[0, N+5:N+5+amountOfClasses,x,y])
                        float bestValue = 0;
                        int bestIndex = 0;
                        for (int i = 0; i < amountOfClasses; i++)
                        {
                            float value = output[0, N + 5 + i, x, y];
                            if (value > bestValue)
                            {
                                bestValue = value;
                                bestIndex = i;
                            }
                        }

                        var tempBox = new Box
                        {
                            x = output[0, N, x, y],
                            y = output[0, N + 1, x, y],
                            width = output[0, N + 2, x, y],
                            height = output[0, N + 3, x, y],
                            label = labels[bestIndex],
                            anchorIndex = anchor + anchorMask,
                            cellIndexY = x,
                            cellIndexX = y
                        };
                        outputBoxList.Add(tempBox);
                    }
                }
            }
        }
        return outputBoxList;
    }

    public List<PixelBox> ConvertBoxToPixelData(List<Box> boxList)
    {
        List<PixelBox> pixelBoxList = new List<PixelBox>();
        for (int i = 0; i < boxList.Count; i++)
        {
            PixelBox tempBox;
            
            //apply anchor mask, each output uses a different preset box
            var boxSections = boxList[i].anchorIndex > 2 ? box20Sections : box40Sections;

            //move marker to the edge of the picture -> move to the center of the cell -> add cell offset (cell size * amount of cells) -> add scale
            tempBox.x = (float)(-inputResolutionX * 0.5) + inputResolutionX / boxSections * 0.5f +
                        inputResolutionX / boxSections * boxList[i].cellIndexX + Sigmoid(boxList[i].x);
            tempBox.y = (float)(-inputResolutionY * 0.5) + inputResolutionX / boxSections * 0.5f +
                          inputResolutionX / boxSections * boxList[i].cellIndexY + Sigmoid(boxList[i].y);

            //select the anchor box and multiply it by scale
            tempBox.width = anchors[boxList[i].anchorIndex * 2] * (float)Math.Pow(Math.E, boxList[i].width);
            tempBox.height = anchors[boxList[i].anchorIndex * 2 + 1] * (float)Math.Pow(Math.E, boxList[i].height);
            tempBox.label = boxList[i].label;
            pixelBoxList.Add(tempBox);
        }
        
        return pixelBoxList;
    }

    public List<PixelBox> NonMaxSuppression(List<PixelBox> boxList)
    {
        for (int i = 0; i < boxList.Count - 1; i++)
        {
            for (int j = i + 1; j < boxList.Count; j++)
            {
                if (IntersectionOverUnion(boxList[i], boxList[j]) > iouThreshold && boxList[i].label == boxList[j].label)
                {
                    boxList.RemoveAt(i);
                }
            }
        }
        return boxList;
    }

    public float IntersectionOverUnion(PixelBox box1, PixelBox box2)
    {
        //top left and bottom right corners of two rectangles
        float b1x1 = box1.x - 0.5f * box1.width;
        float b1x2 = box1.x + 0.5f * box1.width;
        float b1y1 = box1.y - 0.5f * box1.height;
        float b1y2 = box1.y + 0.5f * box1.height;
        float b2x1 = box2.x - 0.5f * box2.width;
        float b2x2 = box2.x + 0.5f * box2.width;
        float b2y1 = box2.y - 0.5f * box2.height;
        float b2y2 = box2.y + 0.5f * box2.height;
        
        //intersection rectangle
        float xLeft = Math.Max(b1x1, b2x1);
        float yTop = Math.Max(b1y1, b2y1);
        float xRight = Math.Max(b1x2, b2x2);
        float yBottom = Math.Max(b1y2, b2y2);
        
        //check if intersection rectangle exist
        if (xRight < xLeft || yBottom < yTop)
        {
            return 0.0f;
        }
        
        float intersectionArea = (xRight - xLeft) * (yBottom - yTop);
        float b1area = (b1x2 - b1x1) * (b1y2 - b1y1);
        float b2area = (b2x2 - b2x1) * (b2y2 - b2y1);
        return intersectionArea / (b1area + b2area - intersectionArea);
    }
    
    public float Sigmoid(float value) {
        return 1.0f / (1.0f + (float) Math.Exp(-value));
    }

    int N = 0;
    public void DrawBox(PixelBox box)
    {
        N++;
        Color color = Color.yellow;// Color.HSVToRGB((N * 0.618f) % 1f, 1f, 1f);
        //add bounding box
        GameObject panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;//new Color(0, 1, 1, 1f);
        img.sprite  = boxTexture;
        

        panel.transform.SetParent(displayLocation, false);
        panel.transform.localPosition = new Vector3(box.x, -box.y);


        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        //add class label
        GameObject text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        Text txt = text.AddComponent<Text>();
        text.transform.SetParent(panel.transform, false);
        txt.text = box.label;
        txt.color = color;
        txt.fontSize = 40;
        txt.font = font;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;
        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.anchorMin = new Vector2(0,0);
        rt2.anchorMax = new Vector2(1, 1);

        img.sprite = boxTexture;
        img.type = Image.Type.Sliced;
    }

    public void ClearAnnotations()
    {
        foreach (Transform child in displayLocation) {
            Destroy(child.gameObject);
        }
    }

    void CleanUp()
    {
        engine?.Dispose();

        foreach (var key in inputs.Keys)
        {
            inputs[key].Dispose();
        }

        inputs.Clear();
    }

    private void OnDestroy()
    {
        CleanUp();
    }
}
