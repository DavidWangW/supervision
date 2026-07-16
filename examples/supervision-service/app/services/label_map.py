"""Map detection class names to Chinese labels for on-video annotation.

The YOLO weights report English class names (e.g. ``car``, ``truck``). This
module translates the common COCO-style names to Chinese so the annotated
result video shows localized labels while it plays in the browser.
"""

from pathlib import Path

from app.config import APP_DIR

# COCO-style class names -> Chinese. Covers the vehicle and common object
# categories produced by the detection models used in this service. Any class
# not present here falls back to its original English name.
CLASS_NAME_TO_CHINESE: dict[str, str] = {
    "person": "行人",
    "people": "人群",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "bus": "公交车",
    "truck": "卡车",
    "train": "火车",
    "boat": "船只",
    "ship": "船只",
    "airplane": "飞机",
    "aeroplane": "飞机",
    "traffic light": "红绿灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车标志",
    "traffic sign": "交通标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skateboard": "滑板",
    "sports ball": "球",
    "kite": "风筝",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
    "rider": "骑行者",
    "license plate": "车牌",
}

# Candidate Chinese-capable font files, in priority order. The first path that
# exists on the host is used so the service works across Windows / Linux / macOS
# as well as when a font is bundled next to the application.
_CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    str(APP_DIR / "fonts" / "simhei.ttf"),
    str(APP_DIR / "fonts" / "msyh.ttc"),
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/PingFang.ttc",
)


def resolve_chinese_font() -> str | None:
    """Return the path to an available Chinese-capable font, or ``None``.

    Returns:
        Absolute path to a ``.ttf``/``.ttc`` font that can render Chinese glyphs,
        or ``None`` when no bundled or system font could be found.
    """
    for candidate in _CHINESE_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def to_chinese_class_name(class_name: str) -> str:
    """Return the Chinese label for a class name, falling back to the original.

    Args:
        class_name: English class name reported by the detection model.

    Returns:
        Chinese label when known, otherwise the original ``class_name``.
    """
    return CLASS_NAME_TO_CHINESE.get(class_name.lower(), class_name)


def build_chinese_labels(class_names: dict[int, str], class_ids) -> list[str]:
    """Build Chinese label strings for a batch of detections.

    Args:
        class_names: Mapping from class id to English name, e.g. ``model.names``.
        class_ids: Iterable of integer class ids for the current detections.

    Returns:
        List of Chinese label strings aligned with ``class_ids``.
    """
    return [to_chinese_class_name(class_names[int(class_id)]) for class_id in class_ids]
