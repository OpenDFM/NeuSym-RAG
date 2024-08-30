#coding=utf8
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
from PIL import Image, ImageDraw, ImageFont

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