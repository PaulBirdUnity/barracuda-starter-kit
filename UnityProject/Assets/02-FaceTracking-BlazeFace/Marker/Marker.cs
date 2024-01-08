using UnityEngine;
using UI = UnityEngine.UI;

namespace MediaPipe.BlazeFace
{

    public sealed class Marker : MonoBehaviour
    {
        public RunInferenceBlazeFace.Detection detection { get; set; }

        [SerializeField] RectTransform[] _keyPoints;

        Marker _marker;
        RectTransform _xform;
        RectTransform _parent;
        UI.Text _label;

        Vector3[] _originalScale = new Vector3[6];

        void SetKeyPoint(RectTransform xform, Vector2 point)
          => xform.anchoredPosition =
               point * _parent.rect.size - _xform.anchoredPosition;

        void Start()
        {
            _marker = GetComponent<Marker>();
            _xform = GetComponent<RectTransform>();
            _parent = (RectTransform)_xform.parent;
            _label = GetComponentInChildren<UI.Text>();
            for (int i = 0; i < 6; i++)
            {
                _originalScale[i] = _keyPoints[i].localScale;
            }
        }

        void LateUpdate()
        {
            var detection = _marker.detection;

            // Bounding box center
            _xform.anchoredPosition = detection.center * _parent.rect.size;

            // Bounding box size
            var size = detection.extent * _parent.rect.size;
            _xform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, size.x);
            _xform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, size.y);

            //float scale = size.y * 0.015f;
            float scale = size.magnitude * 0.01f;

            // Key points
            SetKeyPoint(_keyPoints[0], detection.leftEye);
            SetKeyPoint(_keyPoints[1], detection.rightEye);
            SetKeyPoint(_keyPoints[2], detection.nose);
            SetKeyPoint(_keyPoints[3], detection.mouth);
            SetKeyPoint(_keyPoints[4], detection.leftEar);
            SetKeyPoint(_keyPoints[5], detection.rightEar);

            for (int i = 0; i < 6; i++)
            {
                _keyPoints[i].localScale = _originalScale[i] * scale;
            }

            // Label
            _label.text = $"{(int)(detection.score * 100)}﹪";
        }
    }

}// namespace MediaPipe.BlazeFace
