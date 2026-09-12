from loaders.load_xdf import load_xdf_file


def inspect_streams(xdf_path):
    print(f"\n===== INSPECTING STREAMS: {xdf_path.name} =====")

    streams, _ = load_xdf_file(xdf_path)

    for i, stream in enumerate(streams):
        info = stream["info"]

        name = info.get("name", ["Unknown"])[0]
        stype = info.get("type", ["Unknown"])[0]
        srate = info.get("nominal_srate", ["Unknown"])[0]

        print("\n---------------------------------")
        print(f"Stream Index : {i}")
        print(f"Name         : {name}")
        print(f"Type         : {stype}")
        print(f"SamplingRate : {srate}")
