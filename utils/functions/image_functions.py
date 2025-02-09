#coding=utf8
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
from PIL import Image, ImageDraw, ImageFont
import base64, os

from utils.functions.common_functions import call_llm
from utils.functions.parallel_functions import parallel_write_or_read

def draw_image_with_bbox(
        image_path: str,
        bboxes: List[Tuple[float, float, float, float]],
        output_path: Optional[str] = None,
        bbox_width: int = 1,
        bbox_color: str = 'red',
        add_label: bool = True,
        label_color: str = 'black',
        lable_font: str = 'arial.ttf',
        label_size: int = 18,
        label_position: Optional[Callable[[float, float, str], Tuple[float, float]]] = None
    ) -> str:
    """ Draw the image with bounding boxes and numeric labels.
    @param:
        image_path: str, the path to the image file.
        bboxes: List[Tuple[float, float, float, float]], the list of bounding boxes to draw, (x0, y0, width, height).
        output_path: str, the path to the output image file, default is None, writing to the same image.
        bbox_width: int, the width of the bounding box, default is 1.
        bbox_color: str, the color of the bounding box, default is 'red'.
        add_label: bool, whether to add numeric labels, default is True.
        label_color: str, the color of the numeric label, default is 'black'.
        label_font: str, the font of the numeric label, default is 'arial.ttf'.
        label_size: int, the size of the numeric label, default is 18.
        label_position: Callable[[float, float, float], Tuple[float, float]], where to position the numeric label, a callable function which accepts the position of top-left corner (x, y) and numeric label, and return the (dx, dy) shift. By default, dx = 0, dy = 0.
    @return:
        output_path: str, the path to the output image file.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(lable_font, label_size)
    except IOError:
        font = ImageFont.load_default()

    for idx in range(len(bboxes)):
        x0, y0, w, h = bboxes[idx]
        draw.rectangle([x0, y0, x0 + w, y0 + h], outline=bbox_color, width=bbox_width)
        if add_label:
            text = str(idx)
            if label_position is None:
                label_position = lambda x, y, text: (0, 0)
            dx, dy = label_position(x0, y0, text)
            text_position = (x0 + dx, y0 + dy)
            draw.text(text_position, text, font=font, fill=label_color)

    if output_path is None:
        output_path = image_path
    image.save(output_path)
    return output_path


def get_image_mime_type(image_path: str) -> str:
    """ Get the mime type of the image file according to its extension.
    """
    ext = os.path.basename(image_path).split('.')[-1].lower()
    mime_types = {
        'bmp': 'image/bmp',
        'dib': 'image/bmp',
        'icns': 'image/icns',
        'ico': 'image/x-icon',
        'jfif': 'image/jpeg',
        'jpe': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'j2c': 'image/jp2',
        'j2k': 'image/jp2',
        'jp2': 'image/jp2',
        'jpc': 'image/jp2',
        'jpf': 'image/jp2',
        'jpx': 'image/jp2',
        'apng': 'image/png',
        'png': 'image/png',
        'bw': 'image/sgi',
        'rgb': 'image/sgi',
        'rgba': 'image/sgi',
        'sgi': 'image/sgi',
        'tif': 'image/tiff',
        'tiff': 'image/tiff',
        'webp': 'image/webp',
    }
    return mime_types.get(ext, 'image/jpeg')


def get_image_message(
        template: str,
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None,
        mine_type: str = 'image/jpeg'
    ) -> Dict[str, Any]:
    """ Get the image message for LLM calling.
    @args:
        template: str, the description/instruction for the image.
        image_path: str, path to the image file you want to summary (overwrite `base64_image`).
        base64_image: str, base64 encoded image string. Either `image_path` or `base64_image` must be provided.
        mine_type: str, the mine type of the image, should be specified if only `base64_image` is provided, default to 'image/jpeg'.
    @return:
        message: dict, a role-content message pair
    """
    assert image_path is not None or base64_image is not None, "Either `image_path` or `base64_image` must be provided."
    mine_type = get_image_mime_type(image_path) if image_path is not None else mine_type
    if image_path is not None:
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
    message = {
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': template
            },
            {
                'type': 'image_url',
                "image_url": {
                    "url":  f"data:{mine_type};base64,{base64_image}"
                }
            }
        ]
    }
    
    return message

def get_image_summary(
        image_path: str,
        max_length: int = 50,
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
    """ Get the image summary using LLM.
    @args:
        image_path: str, path to the image file you want to summary.
        max_length: int, the maximum length of the summary, default to 50.
    @return:
        summary: str, the image summary.
    """
    system_template = "You are an intelligent assistant who is expert at summarizing images."
    image_template = f'Please directly return the summary without any extra information or formatting. And you should summarize it in no more than {max_length} words. The image is as follows:'
    
    image_message = get_image_message(image_template, image_path=image_path)
    if kwargs.get("parallel"):
        summary = parallel_write_or_read(
            template=system_template, 
            image=image_message,
            **kwargs
        )
        if summary is not None: return summary
    
    summary = call_llm(
        template=system_template, 
        model=model, 
        top_p=top_p, 
        temperature=temperature,
        image=image_message
    )
    return summary