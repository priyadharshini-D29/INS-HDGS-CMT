import pyxdf


def load_xdf_file(xdf_path):
    streams, header = pyxdf.load_xdf(str(xdf_path))
    return streams, header


def identify_streams(streams):
    identified = {
        "EEG": None,
        "ET": None,
        "MouseButtons": None,
        "MousePosition": None,
        "Markers": None
    }

    for stream in streams:
        info = stream["info"]

        stream_type = info.get("type", [""])[0]
        stream_name = info.get("name", [""])[0]

        if stream_type == "EEG":
            identified["EEG"] = stream
        elif stream_type == "Eyetracker":
            identified["ET"] = stream
        elif stream_name == "MouseButtons":
            identified["MouseButtons"] = stream
        elif stream_name == "MousePosition":
            identified["MousePosition"] = stream
        elif stream_name == "MyMarkerStream3":
            identified["Markers"] = stream

    return identified
