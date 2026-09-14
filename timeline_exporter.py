"""
timeline_exporter.py - Generates Final Cut Pro 7 XML and CMX 3600 EDL timeline project files
compatible with Adobe Premiere Pro, DaVinci Resolve, and Final Cut Pro.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict


def _frames_to_timecode(total_frames: int, fps: float) -> str:
    """Converts integer frames into standard non-drop frame timecode string (HH:MM:SS:FF)."""
    int_fps = int(round(fps)) if int(round(fps)) > 0 else 30
    total_frames = max(0, int(total_frames))
    frames = total_frames % int_fps
    total_seconds = total_frames // int_fps
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def seconds_to_timecode(seconds: float, fps: float) -> str:
    """
    Converts seconds into standard non-drop frame timecode string (HH:MM:SS:FF).
    Handles floating point inaccuracies and standard framerates (23.976, 24, 25, 29.97, 30, 59.94, 60).
    """
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    return _frames_to_timecode(total_frames, fps)


def _format_file_url(path: str) -> str:
    """Formats file path as standard file:// URL for FCP XML."""
    if path.startswith("file://"):
        return path
    if os.path.isabs(path) or path.startswith("/") or path.startswith("\\"):
        norm_path = path.replace("\\", "/")
    else:
        norm_path = os.path.abspath(path).replace("\\", "/")
    if not norm_path.startswith("/"):
        norm_path = "/" + norm_path
    return f"file://localhost{norm_path}"


def generate_fcp7_xml(
    source_media_path: str,
    segments: List[Tuple[float, float]],
    fps: float,
    width: int = 1920,
    height: int = 1080,
    sample_rate: int = 48000,
    sequence_name: str = "Interview_Cut"
) -> str:
    """
    Generates standard FCP7 XML sequence (<xmeml version="4">) containing individual clips
    for kept speech segments with video and stereo audio tracks (A1 and A2).
    """
    timebase = int(round(fps)) if int(round(fps)) > 0 else 30
    ntsc = "TRUE" if abs(fps - 29.97) < 0.01 or abs(fps - 23.976) < 0.01 or abs(fps - 59.94) < 0.01 else "FALSE"

    # Filter valid segments where out_frame > in_frame
    valid_cuts: List[Tuple[int, int]] = []
    for s_sec, e_sec in segments:
        in_f = int(round(s_sec * fps))
        out_f = int(round(e_sec * fps))
        if out_f > in_f:
            valid_cuts.append((in_f, out_f))

    total_duration_frames = sum(out_f - in_f for in_f, out_f in valid_cuts)
    max_out_frame = max([out_f for _, out_f in valid_cuts], default=0)
    master_duration = max(max_out_frame, int(round(999999 * fps)))

    root = ET.Element("xmeml", version="4")
    sequence = ET.SubElement(root, "sequence")
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(total_duration_frames)

    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = ntsc

    timecode = ET.SubElement(sequence, "timecode")
    tc_rate = ET.SubElement(timecode, "rate")
    ET.SubElement(tc_rate, "timebase").text = str(timebase)
    ET.SubElement(tc_rate, "ntsc").text = ntsc
    ET.SubElement(timecode, "string").text = "00:00:00:00"
    ET.SubElement(timecode, "frame").text = "0"

    media = ET.SubElement(sequence, "media")

    # Video track definition
    video = ET.SubElement(media, "video")
    v_format = ET.SubElement(video, "format")
    samplecharacteristics = ET.SubElement(v_format, "samplecharacteristics")
    ET.SubElement(samplecharacteristics, "width").text = str(width)
    ET.SubElement(samplecharacteristics, "height").text = str(height)
    ET.SubElement(samplecharacteristics, "pixelaspectratio").text = "square"
    v_rate = ET.SubElement(samplecharacteristics, "rate")
    ET.SubElement(v_rate, "timebase").text = str(timebase)
    ET.SubElement(v_rate, "ntsc").text = ntsc

    v_track = ET.SubElement(video, "track")

    # Audio tracks (A1 and A2 stereo channels)
    audio = ET.SubElement(media, "audio")
    ET.SubElement(audio, "numOutputChannels").text = "2"
    a_format = ET.SubElement(audio, "format")
    a_samplecharacteristics = ET.SubElement(a_format, "samplecharacteristics")
    ET.SubElement(a_samplecharacteristics, "depth").text = "16"
    ET.SubElement(a_samplecharacteristics, "samplerate").text = str(sample_rate)

    a1_track = ET.SubElement(audio, "track")
    a2_track = ET.SubElement(audio, "track")

    media_filename = os.path.basename(source_media_path)
    file_url = _format_file_url(source_media_path)

    timeline_in = 0
    for idx, (in_frame, out_frame) in enumerate(valid_cuts, start=1):
        clip_duration = out_frame - in_frame
        timeline_out = timeline_in + clip_duration

        # Video clipitem
        v_clip = ET.SubElement(v_track, "clipitem", id=f"clipitem-v-{idx}")
        ET.SubElement(v_clip, "name").text = f"{media_filename} [Part {idx}]"
        ET.SubElement(v_clip, "duration").text = str(master_duration)
        v_clip_rate = ET.SubElement(v_clip, "rate")
        ET.SubElement(v_clip_rate, "timebase").text = str(timebase)
        ET.SubElement(v_clip_rate, "ntsc").text = ntsc
        ET.SubElement(v_clip, "start").text = str(timeline_in)
        ET.SubElement(v_clip, "end").text = str(timeline_out)
        ET.SubElement(v_clip, "in").text = str(in_frame)
        ET.SubElement(v_clip, "out").text = str(out_frame)

        # File element reference
        v_file = ET.SubElement(v_clip, "file", id="file-master")
        ET.SubElement(v_file, "name").text = media_filename
        ET.SubElement(v_file, "pathurl").text = file_url
        v_file_rate = ET.SubElement(v_file, "rate")
        ET.SubElement(v_file_rate, "timebase").text = str(timebase)
        ET.SubElement(v_file_rate, "ntsc").text = ntsc
        ET.SubElement(v_file, "duration").text = str(master_duration)

        v_file_media = ET.SubElement(v_file, "media")
        v_file_video = ET.SubElement(v_file_media, "video")
        v_sample = ET.SubElement(v_file_video, "samplecharacteristics")
        ET.SubElement(v_sample, "width").text = str(width)
        ET.SubElement(v_sample, "height").text = str(height)

        v_file_audio = ET.SubElement(v_file_media, "audio")
        a_sample = ET.SubElement(v_file_audio, "samplecharacteristics")
        ET.SubElement(a_sample, "depth").text = "16"
        ET.SubElement(a_sample, "samplerate").text = str(sample_rate)
        ET.SubElement(v_file_audio, "channelcount").text = "2"

        v_source = ET.SubElement(v_clip, "sourcetrack")
        ET.SubElement(v_source, "mediatype").text = "video"
        ET.SubElement(v_source, "trackindex").text = "1"

        # Audio Track 1 & 2 clipitems
        for a_idx, a_track in [(1, a1_track), (2, a2_track)]:
            a_clip = ET.SubElement(a_track, "clipitem", id=f"clipitem-a{a_idx}-{idx}")
            ET.SubElement(a_clip, "name").text = f"{media_filename} [Part {idx}]"
            ET.SubElement(a_clip, "duration").text = str(master_duration)
            a_clip_rate = ET.SubElement(a_clip, "rate")
            ET.SubElement(a_clip_rate, "timebase").text = str(timebase)
            ET.SubElement(a_clip_rate, "ntsc").text = ntsc
            ET.SubElement(a_clip, "start").text = str(timeline_in)
            ET.SubElement(a_clip, "end").text = str(timeline_out)
            ET.SubElement(a_clip, "in").text = str(in_frame)
            ET.SubElement(a_clip, "out").text = str(out_frame)
            ET.SubElement(a_clip, "file", id="file-master")

            a_source = ET.SubElement(a_clip, "sourcetrack")
            ET.SubElement(a_source, "mediatype").text = "audio"
            ET.SubElement(a_source, "trackindex").text = str(a_idx)

        timeline_in = timeline_out

    rough_string = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def generate_cmx3600_edl(
    source_media_path: str,
    segments: List[Tuple[float, float]],
    fps: float,
    sequence_name: str = "Interview_Cut"
) -> str:
    """Generates standard CMX 3600 Edit Decision List (EDL)."""
    lines = [
        f"TITLE: {sequence_name}",
        "FCM: NON-DROP FRAME",
        ""
    ]

    base_stem = os.path.splitext(os.path.basename(source_media_path))[0]
    clean_stem = "".join(c for c in base_stem if c.isalnum()).upper()
    reel_name = clean_stem[:8] if clean_stem else "AX"
    media_filename = os.path.basename(source_media_path)

    timeline_in_frames = 0
    event_idx = 1

    for start_sec, end_sec in segments:
        in_frame = int(round(start_sec * fps))
        out_frame = int(round(end_sec * fps))
        dur_frames = out_frame - in_frame
        if dur_frames <= 0:
            continue

        timeline_out_frames = timeline_in_frames + dur_frames

        src_in = _frames_to_timecode(in_frame, fps)
        src_out = _frames_to_timecode(out_frame, fps)
        rec_in = _frames_to_timecode(timeline_in_frames, fps)
        rec_out = _frames_to_timecode(timeline_out_frames, fps)

        lines.append(f"{event_idx:03d}  {reel_name:<8} AA/V  C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {media_filename}")
        lines.append("")

        timeline_in_frames = timeline_out_frames
        event_idx += 1

    return "\n".join(lines)


def export_timeline_files(
    source_media_path: str,
    segments: List[Tuple[float, float]],
    fps: float,
    output_base_path: str,
    width: int = 1920,
    height: int = 1080,
    sample_rate: int = 48000,
    sequence_name: str = "Interview_Cut"
) -> Dict[str, str]:
    """Saves both XML and EDL files for NLE import."""
    xml_path = f"{output_base_path}_timeline.xml"
    edl_path = f"{output_base_path}_timeline.edl"

    out_dir = os.path.dirname(os.path.abspath(xml_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    xml_text = generate_fcp7_xml(
        source_media_path=source_media_path,
        segments=segments,
        fps=fps,
        width=width,
        height=height,
        sample_rate=sample_rate,
        sequence_name=sequence_name
    )
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_text)

    edl_text = generate_cmx3600_edl(
        source_media_path=source_media_path,
        segments=segments,
        fps=fps,
        sequence_name=sequence_name
    )
    with open(edl_path, "w", encoding="utf-8") as f:
        f.write(edl_text)

    return {"xml": xml_path, "edl": edl_path}
