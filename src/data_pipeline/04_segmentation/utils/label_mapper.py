LABEL_MAP = {
    "FYLLADIO_1.tif": "ImagePage_1",
    "FYLLADIO_2.tif": "ImagePage_2",
    "FYLLADIO_3.tif": "ImagePage_3",
    "FYLLADIO_4.tif": "ImagePage_4",
    "FYLLADIO_5.tif": "ImagePage_5",
    "FYLLADIO_6.tif": "ImagePage_6",
}


def normalize_event_label(label: str):
    """
    Convert raw marker labels into clean brochure page names.
    """
    for old, new in LABEL_MAP.items():
        if old in label:
            return new
    return label
