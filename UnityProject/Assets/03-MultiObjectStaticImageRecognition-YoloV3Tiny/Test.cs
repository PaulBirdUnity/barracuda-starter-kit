using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public class Test : MonoBehaviour
{
    static Ops ops;
    ITensorAllocator allocator;
    public Texture2D cat;
    // Start is called before the first frame update
    void Start()
    {
        allocator = new TensorCachingAllocator();
        ops = WorkerFactory.CreateOps(Unity.Sentis.BackendType.GPUPixel, allocator);
        Debug.Log("ops=" + ops);

        var A = new TensorFloat(new TensorShape(1, 1, 1, 1), new float[] { 0.3f });
        var B = ops.Square(A);
        B.MakeReadable();
        Debug.Log(B[0]);

        var C = TextureConverter.ToTensor(cat);
        Debug.Log(C.shape);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
