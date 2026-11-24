import base64

def capture():
    import cv2
    camera=cv2.VideoCapture(0)
    try:
        ok,frame=camera.read()
        if not ok: raise RuntimeError('Camera unavailable')
        frame=cv2.resize(frame,(640,480))
        ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
        if not ok: raise RuntimeError('JPEG encoding failed')
        return base64.b64encode(jpeg).decode()
    finally: camera.release()
