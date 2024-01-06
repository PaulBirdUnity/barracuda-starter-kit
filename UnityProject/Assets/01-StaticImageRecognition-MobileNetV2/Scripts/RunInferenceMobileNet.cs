using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;

public class RunInferenceMobileNet : MonoBehaviour
{
    public Texture2D[] inputImage;
    public int selectedImage = 0;
    public int inputResolutionY = 224;
    public int inputResolutionX = 224;
    public RawImage displayImage;
    public ModelAsset srcModel;
    public TextAsset labelsAsset;
    public Text resultClassText;
    public Dropdown backendDropdown;
    
    private string inferenceBackend = "GPUCompute";
    private Model model;
    private IWorker engine;
    private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
    private string[] labels;
    private RenderTexture targetRT;

    static Ops ops;
    ITensorAllocator allocator;

    void Start()
    {
        allocator = new TensorCachingAllocator();

        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;
        AddBackendOptions();
        //parse neural net labels
        labels = labelsAsset.text.Split('\n');
        //load model
        model = ModelLoader.Load(srcModel);
        //format input texture variable
        targetRT = RenderTexture.GetTemporary(inputResolutionX, inputResolutionY, 0, RenderTextureFormat.ARGBHalf);
        //execute inference
        SelectBackendAndExecuteML();
    }

    void SetupEngine()
    {
        ops?.Dispose();
        engine?.Dispose();

        if (inferenceBackend == "CPU")
        {
            engine = WorkerFactory.CreateWorker(BackendType.CPU, model);
            ops = WorkerFactory.CreateOps(BackendType.CPU, allocator);
        }
        else if (inferenceBackend == "GPUCompute")
        {
            engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
            ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);
        }
        else if (inferenceBackend == "PixelShader")
        {
            engine = WorkerFactory.CreateWorker(BackendType.GPUPixel, model);
            ops = WorkerFactory.CreateOps(BackendType.GPUPixel, allocator);
        }
    }

    public void ExecuteML(int imageID)
    {
        selectedImage = imageID;
        displayImage.texture = inputImage[selectedImage];

        //preprocess image for input
        using var input0 = TextureConverter.ToTensor(inputImage[selectedImage], 224, 224, 3);
        using var input = Normalise(input0);
        //execute neural net
        engine.Execute(input);
        //read output tensor
        var output = engine.PeekOutput() as TensorFloat;
        var argmax = ops.ArgMax(output, 1, false);
        argmax.MakeReadable();
        //select the best output class and print the results
        var res = argmax[0];
        var label = labels[res];
        output.MakeReadable();
        var accuracy = output[res];
        resultClassText.text = $"{label} {Math.Round(accuracy*100f, 1)}﹪";
        //clean memory
        Resources.UnloadUnusedAssets();
    }

    TensorFloat Normalise(TensorFloat image)
    {
        using var M = new TensorFloat(new TensorShape(1, 3, 1, 1), new float[]
        {
           1/0.229f, 1/0.224f, 1/0.225f
        });
        using var P = new TensorFloat(new TensorShape(1, 3, 1, 1), new float[]
        {
            0.485f, 0.456f, 0.406f
        });
        using var image2 = ops.Sub(image, P);
        return ops.Mul(image2, M);
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
    }

    public void SelectBackendAndExecuteML()
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
        ExecuteML(selectedImage);
    }
    
    private void OnDestroy()
    {
        engine?.Dispose();

        foreach (var key in inputs.Keys)
        {
            inputs[key]?.Dispose();
        }
		
        inputs.Clear();

        ops?.Dispose();
        allocator?.Dispose();
    }
}
