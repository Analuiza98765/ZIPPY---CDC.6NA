
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    [Header("Centro da casa")]
    public Transform houseCenter;

    [Header("Posicionamento")]
    public float cameraHeight = 22f;
    public float cameraDistance = 8f;

    [Header("Enquadramento")]
    public float cameraSize = 14f;

    private Camera cam;

    void Start()
    {
        cam = GetComponent<Camera>();

        if (houseCenter == null)
        {
            Debug.LogError("Arraste o CameraTarget para House Center!");
            return;
        }

        // Posiciona a camera acima e um pouco afastada
        Vector3 offset = new Vector3(
            0f,
            cameraHeight,
            -cameraDistance
        );

        transform.position = houseCenter.position + offset;

        // Aponta a camera para o centro da casa
        transform.LookAt(houseCenter.position);

        // Configura a visao panoramica
        cam.orthographic = true;
        cam.orthographicSize = cameraSize;

        cam.nearClipPlane = 0.3f;
        cam.farClipPlane = 1000f;
    }
}
