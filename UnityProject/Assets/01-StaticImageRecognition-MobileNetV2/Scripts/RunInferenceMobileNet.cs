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
    public Material preprocessMaterial;
    public Dropdown backendDropdown;
    
    private string inferenceBackend = "CSharpBurst";
    //"ComputePrecompiled";//
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
        ops = WorkerFactory.CreateOps(Unity.Sentis.BackendType.GPUCompute, allocator);
        Debug.Log("ops=" + ops);

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

    public void ExecuteML(int imageID)
    {
        selectedImage = imageID;
        displayImage.texture = inputImage[selectedImage];
        
        if (inferenceBackend == "CSharpBurst")
        {
            engine = WorkerFactory.CreateWorker(BackendType.CPU, model);
        } 
        else if (inferenceBackend == "ComputePrecompiled")
        {
            engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        } 
        else if (inferenceBackend == "PixelShader")
        {
            engine = WorkerFactory.CreateWorker(BackendType.GPUPixel, model);
        }

        //preprocess image for input
        var input =  TextureConverter.ToTensor(PrepareTextureForInput(inputImage[selectedImage]), 224, 224, 3);
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
        input.Dispose();
        engine.Dispose();
        Resources.UnloadUnusedAssets();
    }

    Texture PrepareTextureForInput(Texture2D src)
    {
        RenderTexture.active = targetRT;
        //normalization is applied in the NormalizeInput shader
        Graphics.Blit(src, targetRT, preprocessMaterial);

        var  result = new Texture2D(targetRT.width, targetRT.height, TextureFormat.RGBAHalf, false);
        result.ReadPixels(new Rect(0,0, targetRT.width, targetRT.height), 0, 0);
        result.Apply();
        return result;
    }

    public void AddBackendOptions()
    {
        List<string> options = new List<string> ();
        options.Add("CSharpBurst");
        #if !UNITY_WEBGL
        options.Add("ComputePrecompiled");
        #endif
        options.Add("PixelShader");
        backendDropdown.ClearOptions ();
        backendDropdown.AddOptions(options);
    }

    public void SelectBackendAndExecuteML()
    {
        
        if (backendDropdown.options[backendDropdown.value].text == "CSharpBurst")
        {
            inferenceBackend = "CSharpBurst";
        }
        else if (backendDropdown.options[backendDropdown.value].text == "ComputePrecompiled")
        {
            inferenceBackend = "ComputePrecompiled";
        }
        else if (backendDropdown.options[backendDropdown.value].text == "PixelShader")
        {
            inferenceBackend = "PixelShader";
        }
        ExecuteML(selectedImage);
    }
    
    private void OnDestroy()
    {
        engine?.Dispose();

        foreach (var key in inputs.Keys)
        {
            inputs[key].Dispose();
        }
		
        inputs.Clear();

        ops?.Dispose();
        allocator?.Dispose();
    }
}
