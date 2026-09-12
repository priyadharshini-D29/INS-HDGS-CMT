import pyxdf


def load_xdf_file(xdf_path):

    streams, header = pyxdf.load_xdf(xdf_path)

    identified = {}

    for stream in streams:

        name = stream["info"]["name"][0]

        if "EEG" in name or "WS" in name:
            identified["EEG"] = stream

        elif "Tobii" in name:
            identified["ET"] = stream

        elif "MousePosition" in name:
            identified["MousePosition"] = stream

        elif "MouseButtons" in name:
            identified["MouseButtons"] = stream

        elif "Marker" in name:
            identified["Markers"] = stream

    return identified
