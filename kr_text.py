#opencv에서 한글을 지원하지 않기에, pillow를 이용하기로 하였다.

import functools
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",              # 맑은 고딕 (Windows 기본 한글 폰트)
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux (나눔고딕, 설치돼 있으면)
]


@functools.lru_cache(maxsize=None)
def _load_font(px_size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, px_size)
    return ImageFont.load_default()


def _is_ascii(text):
    return all(ord(ch) < 128 for ch in text)


def put_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, center=False):
    """cv2.putText 대체 함수. 텍스트에 한글이 섞여 있으면 Pillow로, 순수 ASCII면
    기존과 동일하게 cv2로 그린다. img는 in-place로 수정되고 그대로 반환된다."""
    lineType = cv2.LINE_AA if lineType is None else lineType

    if _is_ascii(text):
        draw_org = org
        if center:
            (tw, th), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
            draw_org = (int(org[0] - tw / 2), int(org[1] + th / 2))
        cv2.putText(img, text, draw_org, fontFace, fontScale, color, thickness, lineType)
        return img

    px_size = max(10, int(round(fontScale * 34)))
    font = _load_font(px_size)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = org
    if center:
        x = int(x - tw / 2) - bbox[0]
        y = int(y - th / 2) - bbox[1]
    else:
        # cv2.putText's org is the bottom-left baseline; approximate that here.
        x = x - bbox[0]
        y = y - px_size - bbox[1]

    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img
