using UnityEngine;
using UI = UnityEngine.UI;
using System.Collections.Generic;
using Unity.Sentis;
using System.Runtime.InteropServices;
using UnityEngine.Android;

namespace MediaPipe.BlazeFace
{

    public sealed class RunInferenceBlazeFace : MonoBehaviour
    {
        private string _deviceName = "";
        [SerializeField] Vector2Int _resolution = new Vector2Int(1080, 1080);

        WebCamTexture _webcam;
        RenderTexture _buffer;

        bool _useWebcam = !true;

        public Texture2D _testFace;

        private Texture2D _image = null;
        [SerializeField, Range(0, 1)] float _threshold = 0.75f;
        [SerializeField] UI.RawImage _previewUI = null;
        [SerializeField] Marker _markerPrefab = null;

        Marker[] _markers = new Marker[16];

        // Maximum number of detections. This value must be matched with
        // MAX_DETECTION in Common.hlsl.
        const int MaxDetection = 64;

        public ModelAsset _modelAsset;
        public ComputeShader _postprocess1;
        public ComputeShader _postprocess2;
        ComputeBuffer _post1Buffer;
        ComputeBuffer _post2Buffer;
        ComputeBuffer _countBuffer;
        IWorker _worker;
        int _size;

        public GameObject webglWarning;
        public UI.Dropdown _cameraDropdown;
        public UI.Dropdown _backendDropdown;

        static Ops ops;
        ITensorAllocator allocator;

        Model _model;

        //
        // Detection structure. The layout of this structure must be matched with
        // the one defined in Common.hlsl.
        //
        [StructLayout(LayoutKind.Sequential)]
        public readonly struct Detection
        {
            // Bounding box
            public readonly Vector2 center;
            public readonly Vector2 extent;

            // Key points
            public readonly Vector2 leftEye;
            public readonly Vector2 rightEye;
            public readonly Vector2 nose;
            public readonly Vector2 mouth;
            public readonly Vector2 leftEar;
            public readonly Vector2 rightEar;

            // Confidence score [0, 1]
            public readonly float score;

            // Padding
            public readonly float pad1, pad2, pad3;

            // sizeof(Detection)
            public const int Size = 20 * sizeof(float);
        };

        void Start()
        {
            allocator = new TensorCachingAllocator();
            // Prepare camera
#if PLATFORM_ANDROID
            Permission.RequestUserPermission(Permission.Camera);
            while (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                
            }
#endif
#if UNITY_IOS
            Application.RequestUserAuthorization(UserAuthorization.WebCam);
            while (!Application.HasUserAuthorization(UserAuthorization.WebCam))
            {
                
            }
#endif
#if UNITY_WEBGL
            webglWarning.SetActive(true);
#endif

            if (_useWebcam)
            {
                _webcam = new WebCamTexture(_deviceName, _resolution.x, _resolution.y);
                _webcam.requestedFPS = 30;
                _webcam.Play();
                _buffer = new RenderTexture(_resolution.x, _resolution.y, 0);
            }
            else
            {
                _buffer = new RenderTexture(_resolution.x, _resolution.y, 0);
                Graphics.Blit(_testFace, _buffer);
            }

            Screen.orientation = ScreenOrientation.LandscapeLeft;

            AddCameraOptions();
            AddBackendOptions();

            //initialization
            SetupModel();
            SetupEngine();

            // Marker population
            for (var i = 0; i < _markers.Length; i++)
                _markers[i] = Instantiate(_markerPrefab, _previewUI.transform);

            // Static image test: Run the detector once.
            if (_image != null) runInference(_image);
        }

        void Update()
        {
            if (_useWebcam)
            {
                // Format video inpout
                if (!_webcam.didUpdateThisFrame) return;

                var aspect1 = (float)_webcam.width / _webcam.height;
                var aspect2 = (float)_resolution.x / _resolution.y;
                var gap = aspect2 / aspect1;

                var vflip = _webcam.videoVerticallyMirrored;
                var scale = new Vector2(gap, vflip ? -1 : 1);
                var offset = new Vector2((1 - gap) / 2, vflip ? 1 : 0);

                Graphics.Blit(_webcam, _buffer, scale, offset);
            }

        }
        void LateUpdate()
        {
            // Webcam test: Run the detector every frame.
            runInference(_buffer);
        }

        public void AddCameraOptions()
        {
            List<string> options = new List<string>();
            foreach (var option in WebCamTexture.devices)
            {
                options.Add(option.name);
            }
            _cameraDropdown.ClearOptions();
            _cameraDropdown.AddOptions(options);
        }

        public void AddBackendOptions()
        {
            List<string> options = new List<string>();
#if !UNITY_WEBGL
            options.Add("GPUCompute");
#endif
            options.Add("PixelShader"); //not supported with this demo
            options.Add("CPU"); //not supported with this demo
            _backendDropdown.ClearOptions();
            _backendDropdown.AddOptions(options);
        }

        public void SwapCamera()
        {
            if (_useWebcam)
            {
                _webcam.Stop();
                _webcam.deviceName = _cameraDropdown.options[_cameraDropdown.value].text;
                _webcam.Play();
            }
        }

        public void ProcessImage(Texture image, float threshold = 0.75f)
          => ExecuteML(image, threshold);


        void SetupModel()
        {
            _model = ModelLoader.Load(_modelAsset);
            _size = _model.inputs[0].shape.ToTensorShape()[1]; // Input tensor width

            _post1Buffer = new ComputeBuffer(MaxDetection, Detection.Size, ComputeBufferType.Append);

            _post2Buffer = new ComputeBuffer(MaxDetection, Detection.Size, ComputeBufferType.Append);

            _countBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Raw);
        }
        public void SetupEngine()
        {

            ops?.Dispose();
            _worker?.Dispose();

            if (_backendDropdown.options[_backendDropdown.value].text == "GPUCompute")
            {
                _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, _model);

                ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);
            }
            else if (_backendDropdown.options[_backendDropdown.value].text == "PixelShader")
            {
                _worker = WorkerFactory.CreateWorker(BackendType.GPUPixel, _model);

                ops = WorkerFactory.CreateOps(BackendType.GPUPixel, allocator);
            }
            else if (_backendDropdown.options[_backendDropdown.value].text == "CPU")
            {
                _worker = WorkerFactory.CreateWorker(BackendType.CPU, _model);

                ops = WorkerFactory.CreateOps(BackendType.CPU, allocator);
            }
            
           /* _model.AddLayer(new Unity.Sentis.Layers.NonMaxSuppression("NM1", "Identity_2", "Identity"));
            _model.AddLayer(new Unity.Sentis.Layers.NonMaxSuppression("NM2", "Identity_3", "Identity_1"));
            _model.AddOutput("NM1");
            _model.AddOutput("NM2");*/
        }

        

        void ExecuteML(Texture source, float threshold)
        {
            // Reset the compute buffer counters.
            _post1Buffer.SetCounterValue(0);
            _post2Buffer.SetCounterValue(0);


            var transform = new TextureTransform();
            transform.SetDimensions(_size, _size, 3);
            transform.SetTensorLayout(0, 3, 1, 2);
            using var image0 = TextureConverter.ToTensor(source, transform);

            // Pre-process the image to make input in range (-1..1)
            using var image = ops.Mad(image0, 2f, -1f);
            _worker.Execute(image);

            //var NM1 = _worker.PeekOutput("NM1") as TensorFloat;
            //Debug.Log("NM1 shape=" + NM1.shape);
            
            //---- The rest of this gets the output and puts it into rendertextures to execute compute shaders on them
            //-----We should be able to do the same thing using Sentis Tensors

            // Output tensors -> Temporary render textures
            var scores1RT = _worker.CopyOutputToTempRT("Identity", 1, 512);
            var scores2RT = _worker.CopyOutputToTempRT("Identity_1", 1, 384);
            var boxes1RT = _worker.CopyOutputToTempRT("Identity_2", 16, 512);
            var boxes2RT = _worker.CopyOutputToTempRT("Identity_3", 16, 384);

            

            // 1st postprocess (bounding box aggregation)
            AggregateBoundingBoxes(scores1RT, boxes1RT,  0);
            AggregateBoundingBoxes(scores2RT, boxes2RT,  1);

            // Release the temporary render textures.
            RenderTexture.ReleaseTemporary(scores1RT);
            RenderTexture.ReleaseTemporary(scores2RT);
            RenderTexture.ReleaseTemporary(boxes1RT);
            RenderTexture.ReleaseTemporary(boxes2RT);

            // Retrieve the bounding box count.
            ComputeBuffer.CopyCount(_post1Buffer, _countBuffer, 0);

            // 2nd postprocess (overlap removal)
            RemoveOverlappingBoxes();

            // Retrieve the bounding box count after removal.
            ComputeBuffer.CopyCount(_post2Buffer, _countBuffer, 0);

            // Read cache invalidation
            _post2ReadCache = null;
        }

        void AggregateBoundingBoxes(RenderTexture scoresRT, RenderTexture boxesRT, int dispatch) {
            var post1 = _postprocess1;
            post1.SetFloat("_ImageSize", _size);
            post1.SetFloat("_Threshold", _threshold);

            post1.SetTexture(dispatch, "_Scores", scoresRT);
            post1.SetTexture(dispatch, "_Boxes", boxesRT);
            post1.SetBuffer(dispatch, "_Output", _post1Buffer);
            post1.Dispatch(dispatch, 1, 1, 1);
        }

        void RemoveOverlappingBoxes() {
            var post2 = _postprocess2;
            post2.SetBuffer(0, "_Input", _post1Buffer);
            post2.SetBuffer(0, "_Count", _countBuffer);
            post2.SetBuffer(0, "_Output", _post2Buffer);
            post2.Dispatch(0, 1, 1, 1);
        }

        void runInference(Texture input)
        {
            // Face detection
            ProcessImage(input, _threshold);

            // Marker update
            var i = 0;

            foreach (var detection in Detections)
            {
                if (i == _markers.Length) break;
                var marker = _markers[i++];
                marker.detection = detection;
                marker.gameObject.SetActive(true);
            }

            for (; i < _markers.Length; i++)
                _markers[i].gameObject.SetActive(false);

            // UI update
            _previewUI.texture = input;
        }

        public IEnumerable<Detection> Detections => _post2ReadCache ?? UpdatePost2ReadCache();

        Detection[] _post2ReadCache;
        int[] _countReadCache = new int[1];

        Detection[] UpdatePost2ReadCache()
        {
            _countBuffer.GetData(_countReadCache, 0, 0, 1);
            var count = _countReadCache[0];

            _post2ReadCache = new Detection[count];
            _post2Buffer.GetData(_post2ReadCache, 0, 0, count);

            return _post2ReadCache;
        }

        void OnDestroy()
        {
            if(_useWebcam) Destroy(_webcam);
            Destroy(_buffer);
            Dispose();
        }

        public void Dispose()
            => DeallocateObjects();

        void DeallocateObjects()
        {

            _post1Buffer?.Dispose();
            _post1Buffer = null;

            _post2Buffer?.Dispose();
            _post2Buffer = null;

            _countBuffer?.Dispose();
            _countBuffer = null;

            _worker?.Dispose();
            _worker = null;
        }

    }

    static class IWorkerExtensions
    {
        //
        // Retrieves an output tensor from a NN worker and returns it as a
        // temporary render texture. The caller must release it using
        // RenderTexture.ReleaseTemporary.
        //
        public static RenderTexture CopyOutputToTempRT(this IWorker worker, string name, int w, int h)
        {          
            var fmt = RenderTextureFormat.RFloat;
            var rt = RenderTexture.GetTemporary(w, h, 0, fmt);
            var output = worker.PeekOutput(name) as TensorFloat;
            var tensor = output.ShallowReshape(new TensorShape(1, 1, h, w)) as TensorFloat;
            TextureConverter.RenderToTexture(tensor, rt);
            return rt;
        }
    }

} // namespace MediaPipe.BlazeFace
