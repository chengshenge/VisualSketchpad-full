from gradio_client import Client
from multimodal_conversable_agent import MultimodalConversableAgent
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import time, os
import cv2, json, os, sys, time, random
import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import Normalize

# Load all vision experts
from config import SOM_ADDRESS, GROUNDING_DINO_ADDRESS, DEPTH_ANYTHING_ADDRESS


# --------------------------------------------------------------------------------------
# gradio_client compatibility (handle_file vs file vs raw path) + JSON payload loader
# --------------------------------------------------------------------------------------
try:
    # Newer gradio_client
    from gradio_client import handle_file as _handle_file  # type: ignore
except Exception:
    _handle_file = None  # type: ignore

try:
    # Older gradio_client
    from gradio_client import file as _file  # type: ignore
except Exception:
    _file = None  # type: ignore


def _wrap_file(path: str):
    """Wrap local file path for gradio_client across versions."""
    if _handle_file is not None:
        return _handle_file(path)
    if _file is not None:
        return _file(path)
    return path


def _maybe_load_json_payload(x):
    """If x is a JSON filepath string, load it; otherwise return x as-is."""
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str) and x.lower().endswith(".json") and os.path.exists(x):
        with open(x, "r", encoding="utf-8") as f:
            return json.load(f)
    return x


def _raise_if_payload_error(payload, where: str):
    """If server returned an error payload, raise with traceback."""
    if isinstance(payload, dict) and payload.get("error"):
        tb = payload.get("traceback") or ""
        raise RuntimeError(f"{where} failed: {payload.get('error')}\n{tb}")


# Set of Marks
som_client = Client(SOM_ADDRESS)

# Grounding DINO
gd_client = Client(GROUNDING_DINO_ADDRESS)

# DepthAnything
da_client = Client(DEPTH_ANYTHING_ADDRESS)



class AnnotatedImage:
    # A class to represent an annotated image. It contains the annotated image and the original image.
    
    def __init__(self, annotated_image: Image.Image, original_image: Image.Image=None):
        self.annotated_image = annotated_image
        self.original_image = original_image



def segment_and_mark(image, granularity:float = 1.8, alpha:float = 0.1, anno_mode:list = ['Mask', 'Mark']):
    """Use a segmentation model to segment the image, and add colorful masks on the segmented objects. Each segment is also labeled with a number.
    The annotated image is returned along with the bounding boxes of the segmented objects.

    Args:
        image (PIL.Image.Image): The input image.
        granularity (float): The granularity of the segmentation. A larger value indicates more fine-grained segmentation.
        alpha (float): The transparency of the masks.
        anno_mode (list): The annotation mode. 'Mask' indicates to overlay masks on the image. 'Mark' indicates to overlay bounding boxes on the image.

    Returns:
        output_image (AnnotatedImage): The annotated image.
        bboxes (list): The bounding boxes of the segmented objects in normalized coordinates. Each bounding box is represented as (x1, y1, x2, y2).

    Example:
        >>> from PIL import Image
        >>> from tools import segment_and_mark
        >>> image = Image.open("example.jpg")
        >>> output_image, bboxes = segment_and_mark(image)
        >>> output_image.annotated_image.show()
    """
    
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, "JPEG")
        image = tmp_file.name
    
        outputs = som_client.predict(_wrap_file(image), granularity, alpha, "Number", anno_mode)
    
        original_image = Image.open(image)
        if outputs and outputs[0] is not None:
            output_image = Image.open(outputs[0])
        else:
            output_image = original_image.copy()
        output_image = AnnotatedImage(output_image, original_image)
    
        w, h = output_image.annotated_image.size
    
        masks_payload = _maybe_load_json_payload(outputs[1] if isinstance(outputs, (list, tuple)) and len(outputs) > 1 else None)
        _raise_if_payload_error(masks_payload, "segment_and_mark")
        masks = masks_payload
        if isinstance(masks_payload, dict):
            masks = masks_payload.get('masks') or masks_payload.get('results') or masks_payload.get('segments') or masks_payload.get('data')
    
        bboxes = []
    
        for mask in (masks or []):
            if not isinstance(mask, dict) or 'bbox' not in mask:
                continue
            bbox = mask['bbox']
            bboxes.append((bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h))
    
    return output_image, bboxes



def detection(image, objects:list, box_threshold:float = 0.25, text_threshold:float = 0.15):
    """Use a object detection model to detect objects in the image.

    Args:
        image (PIL.Image.Image): The input image.
        objects (list): The list of objects to detect.
        box_threshold (float): The confidence threshold for bounding boxes.
        text_threshold (float): The confidence threshold for text.

    Returns:
        output_image (AnnotatedImage): The annotated image (annotated_image + original_image).
        processed_boxes (list): The bounding boxes of the detected objects in normalized coordinates (x, y, w, h).

    Example:
        >>> from PIL import Image
        >>> from tools import detection
        >>> image = Image.open("example.jpg")
        >>> output_image, boxes = detection(image, ["cat", "dog"])
        >>> output_image.annotated_image.show()
    """
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, "JPEG")
        image = tmp_file.name
    
        outputs = gd_client.predict(_wrap_file(image), ', '.join(objects), box_threshold, text_threshold)
    
        original_image = Image.open(image)
        if outputs and outputs[0] is not None:
            try:
                output_image = Image.open(outputs[0])
            except Exception:
                output_image = original_image.copy()
        else:
            output_image = original_image.copy()
        output_image = AnnotatedImage(output_image, original_image)
    
        payload = _maybe_load_json_payload(outputs[1] if isinstance(outputs, (list, tuple)) and len(outputs) > 1 else None)
        _raise_if_payload_error(payload, "detection")
        boxes = None
        if isinstance(payload, dict):
            boxes = payload.get('boxes_cxcywh_01') or payload.get('boxes')
        elif isinstance(payload, list):
            boxes = payload
        boxes = boxes or []
    
        processed_boxes = []
        for box in boxes:
            processed_boxes.append((box[0] - box[2]/2, box[1] - box[3]/2, box[2], box[3]))
    
    return output_image, processed_boxes



def depth(image: Image.Image):
    """Use a depth estimation model to estimate the depth of the image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        output_image (PIL.Image.Image): The estimated depth image.
    """
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, "JPEG")
        image = tmp_file.name
    
        outputs = da_client.predict(_wrap_file(image))
    
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        outputs = outputs[0]
    output_image = Image.open(outputs)
    
    return output_image



def crop_image(image, x:float, y:float, width:float, height:float):
    """Crop the image based on the bounding box.
    
    Args:
        image (PIL.Image.Image): The input image.
        x (float): The x-coordinate of the top-left corner of the bounding box in normalized coordinates.
        y (float): The y-coordinate of the top-left corner of the bounding box in normalized coordinates.
        width (float): The width of the bounding box in normalized coordinates.
        height (float): The height of the bounding box in normalized coordinates.
    
    Returns:
        cropped_img (PIL.Image.Image): The cropped image.
    """
    w, h = image.size

    x = min(max(0, x), 1)
    y = min(max(0, y), 1)
    x2 = min(max(0, x + width), 1)
    y2 = min(max(0, y + height), 1)

    cropped_img = image.crop((x*w, y*h, x2*w, y2*h))
    return cropped_img




def zoom_in_image_by_bbox(image, box, padding=0.05):
    """Crop the image based on the bounding box with padding."""
    assert padding >= 0.05, "The padding should be at least 0.05"
    x, y, w, h = box
    x, y, w, h = x - padding, y - padding, w + 2*padding, h + 2*padding
    cropped_img = crop_image(image, x, y, w, h)
    return cropped_img



def sliding_window_detection(image, objects):
    """Use a sliding window to detect objects in the image.

    Returns:
        possible_patches (list): A list of annotated image patches that contain detections.
        possible_boxes (list): A list of lists of boxes for each patch.
    """
    
    def check_if_box_margin(box, margin=0.005):
        x_margin = min(box[0], 1 - box[0] - box[2])
        y_margin = min(box[1], 1 - box[1] - box[3])
        return x_margin < margin or y_margin < margin
    
    
    box_width = 1/3
    box_height = 1/3
    
    possible_patches = []
    possible_boxes = []
    
    for x in np.arange(0, 7/9, 2/9):
        for y in np.arange(0, 7/9, 2/9):
            cropped_img = crop_image(image, x, y, box_width, box_height)
            annotated_img, detection_boxes = detection(cropped_img, objects)
        
            margin_flag = True
            for box in detection_boxes:
                if not check_if_box_margin(box):
                    margin_flag = False
                    break
            
            if len(detection_boxes) != 0 and not margin_flag:
                possible_patches.append(annotated_img)
                possible_boxes.append(detection_boxes)
                
    
    return possible_patches, possible_boxes



def overlay_images(background_img, overlay_img, alpha=0.3, bounding_box=[0, 0, 1, 1]):
    """
    Overlay an image onto another image with transparency.
    
    Args:
        background_img (PIL.Image.Image): The background image.
        overlay_img (PIL.Image.Image): The overlay image.
        alpha (float): Transparency level for overlay (0 to 1).
        bounding_box (list): Coordinates and dimensions [x, y, w, h] in normalized form.
    
    Returns:
        PIL.Image.Image: The combined image.
    """
    # Convert normalized bounding box to pixel values
    bg_width, bg_height = background_img.size
    x = int(bounding_box[0] * bg_width)
    y = int(bounding_box[1] * bg_height)
    w = int(bounding_box[2] * bg_width)
    h = int(bounding_box[3] * bg_height)
    
    # Resize overlay image to fit bounding box size
    overlay_resized = overlay_img.resize((w, h), Image.Resampling.LANCZOS)
    
    # Create an alpha mask for transparency
    overlay_with_alpha = overlay_resized.copy()
    overlay_with_alpha.putalpha(int(255 * alpha))
    
    # Paste overlay image onto background image with transparency
    new_img = Image.new("RGBA", background_img.size, (255, 255, 255, 255))
    new_img.paste(background_img, (0, 0))
    new_img.paste(overlay_with_alpha, (x, y, x + w, y + h), overlay_with_alpha)
    
    return new_img.convert("RGB")


# ===== VSK_DEBUG_TOOLS_WRAPPER =====
import os as _os, time as _time

def _vsk_debug_log(_msg: str):
    try:
        with open("/tmp/vsk_tools_called.log", "a", encoding="utf-8") as _f:
            _f.write(f"{_time.strftime('%Y-%m-%d %H:%M:%S')} {_msg}\n")
    except Exception:
        pass

if _os.environ.get("VSK_DEBUG_TOOLS", "0") == "1":
    _vsk_debug_log(f"tools module loaded from: {__file__}")

    # wrap detection
    _orig_detection = detection
    def detection(*args, **kwargs):
        objs = None
        if len(args) >= 2:
            objs = args[1]
        if "objects" in kwargs:
            objs = kwargs["objects"]
        _vsk_debug_log(f"detection called objects={objs}")
        out = _orig_detection(*args, **kwargs)
        try:
            boxes = out[1]
            _vsk_debug_log(f"detection returned num_boxes={len(boxes)} first={boxes[:1]}")
        except Exception as e:
            _vsk_debug_log(f"detection returned (parse boxes failed): {e}")
        return out

    # wrap segment_and_mark (SOM)
    if "segment_and_mark" in globals():
        _orig_segment_and_mark = segment_and_mark
        def segment_and_mark(*args, **kwargs):
            _vsk_debug_log("segment_and_mark called")
            out = _orig_segment_and_mark(*args, **kwargs)
            try:
                boxes = out[1]
                _vsk_debug_log(f"segment_and_mark returned num_boxes={len(boxes)} first={boxes[:1]}")
            except Exception as e:
                _vsk_debug_log(f"segment_and_mark returned (parse boxes failed): {e}")
            return out

    # wrap depth (DepthAnything)
    if "depth" in globals():
        _orig_depth = depth
        def depth(*args, **kwargs):
            _vsk_debug_log("depth called")
            out = _orig_depth(*args, **kwargs)
            try:
                _vsk_debug_log(f"depth returned type={type(out)}")
            except Exception:
                pass
            return out
# ===== END VSK_DEBUG_TOOLS_WRAPPER =====



# ---- External patch injection (tools) ----
def _vsk_autoload_external_tool_patches():
    try:
        from vsk_patches.loader import load_active_patches, register_tool_patches
        enabled = load_active_patches(os.environ.get("VSK_PATCH_CONFIG", "configs/active_patches.json"))
        register_tool_patches(
            tools_module=sys.modules[__name__],
            patch_root=os.environ.get("VSK_PATCH_ROOT", "generated_patches"),
            enabled_tools=enabled.get("tools", []),
        )
    except Exception:
        pass

try:
    _vsk_autoload_external_tool_patches()
except Exception:
    pass
# ---- End external patch injection ----
