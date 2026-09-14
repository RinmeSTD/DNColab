import os
import sys
import xml.etree.ElementTree as ET
import pytest

# Ensure project root is in sys.path when running pytest directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timeline_exporter import (
    seconds_to_timecode,
    generate_fcp7_xml,
    generate_cmx3600_edl,
    export_timeline_files,
)


def test_seconds_to_timecode():
    # Test zero seconds
    assert seconds_to_timecode(0.0, 30.0) == "00:00:00:00"
    assert seconds_to_timecode(0.0, 24.0) == "00:00:00:00"
    assert seconds_to_timecode(0.0, 25.0) == "00:00:00:00"
    assert seconds_to_timecode(0.0, 29.97) == "00:00:00:00"
    assert seconds_to_timecode(0.0, 23.976) == "00:00:00:00"
    assert seconds_to_timecode(0.0, 59.94) == "00:00:00:00"
    assert seconds_to_timecode(0.0, 60.0) == "00:00:00:00"

    # Test fractional seconds
    assert seconds_to_timecode(1.5, 30.0) == "00:00:01:15"
    assert seconds_to_timecode(0.5, 24.0) == "00:00:00:12"
    assert seconds_to_timecode(0.04, 25.0) == "00:00:00:01"
    assert seconds_to_timecode(0.1, 60.0) == "00:00:00:06"
    assert seconds_to_timecode(65.0, 25.0) == "00:01:05:00"

    # Test floating point inaccuracies
    assert seconds_to_timecode(1.0 - 1e-9, 30.0) == "00:00:01:00"
    assert seconds_to_timecode(1.0 + 1e-9, 30.0) == "00:00:01:00"
    assert seconds_to_timecode(1.4999999999, 30.0) == "00:00:01:15"

    # Test > 1 hour
    assert seconds_to_timecode(3600.0, 24.0) == "01:00:00:00"
    assert seconds_to_timecode(3600.0, 60.0) == "01:00:00:00"
    assert seconds_to_timecode(3661.5, 30.0) == "01:01:01:15"
    assert seconds_to_timecode(7325.0, 25.0) == "02:02:05:00"

    # Test NTSC rates timecodes
    assert seconds_to_timecode(10.0, 29.97) == "00:00:10:00"
    assert seconds_to_timecode(10.0, 23.976) == "00:00:10:00"

    # Test negative seconds clamp to 00:00:00:00
    assert seconds_to_timecode(-5.0, 30.0) == "00:00:00:00"


def test_generate_fcp7_xml_validity():
    segments = [(1.0, 5.0), (10.0, 15.0)]
    source_media = "/path/to/interview.mp4"
    xml_content = generate_fcp7_xml(
        source_media_path=source_media,
        segments=segments,
        fps=30.0,
        width=1920,
        height=1080,
        sample_rate=48000,
        sequence_name="Interview_Cut"
    )

    # 1. Parse XML with ElementTree
    root = ET.fromstring(xml_content)
    assert root.tag == "xmeml"
    assert root.attrib.get("version") == "4"

    # 2. Sequence metadata
    sequence = root.find("sequence")
    assert sequence is not None
    assert sequence.find("name").text == "Interview_Cut"
    # Seg 1: 4s = 120 frames. Seg 2: 5s = 150 frames. Total = 270 frames.
    assert sequence.find("duration").text == "270"

    # 3. Video track and clipitem count
    v_track = sequence.find(".//media/video/track")
    assert v_track is not None
    v_clipitems = v_track.findall("clipitem")
    assert len(v_clipitems) == 2

    # Verify first video clipitem in/out arithmetic
    # Seg 1: (1.0, 5.0) -> in=30, out=150, start=0, end=120
    c1 = v_clipitems[0]
    assert c1.find("in").text == "30"
    assert c1.find("out").text == "150"
    assert c1.find("start").text == "0"
    assert c1.find("end").text == "120"
    # File link pointing to source_media_path
    f1 = c1.find("file")
    assert f1 is not None
    assert f1.find("pathurl") is not None
    assert "interview.mp4" in f1.find("pathurl").text

    # Verify second video clipitem
    # Seg 2: (10.0, 15.0) -> in=300, out=450, start=120, end=270
    c2 = v_clipitems[1]
    assert c2.find("in").text == "300"
    assert c2.find("out").text == "450"
    assert c2.find("start").text == "120"
    assert c2.find("end").text == "270"

    # 4. Audio tracks (A1 and A2 stereo channels)
    audio = sequence.find(".//media/audio")
    assert audio is not None
    assert audio.find("numOutputChannels").text == "2"
    a_sample = audio.find(".//format/samplecharacteristics")
    assert a_sample is not None
    assert a_sample.find("samplerate").text == "48000"
    assert a_sample.find("depth").text == "16"

    # Video format
    v_sample = sequence.find(".//media/video/format/samplecharacteristics")
    assert v_sample is not None
    assert v_sample.find("width").text == "1920"
    assert v_sample.find("height").text == "1080"
    assert sequence.find("rate/ntsc").text == "FALSE"

    a_tracks = audio.findall("track")
    assert len(a_tracks) == 2

    for a_track in a_tracks:
        a_clipitems = a_track.findall("clipitem")
        assert len(a_clipitems) == 2
        # Check in/out frame consistency with video
        assert a_clipitems[0].find("in").text == "30"
        assert a_clipitems[0].find("out").text == "150"
        assert a_clipitems[0].find("start").text == "0"
        assert a_clipitems[0].find("end").text == "120"

        assert a_clipitems[1].find("in").text == "300"
        assert a_clipitems[1].find("out").text == "450"
        assert a_clipitems[1].find("start").text == "120"
        assert a_clipitems[1].find("end").text == "270"

        # Verify file link
        assert a_clipitems[0].find("file") is not None


def test_generate_fcp7_xml_empty_and_invalid_segments():
    # Empty segments
    xml_empty = generate_fcp7_xml("/path/to/empty.mp4", [], fps=30.0)
    root_empty = ET.fromstring(xml_empty)
    assert root_empty.tag == "xmeml"
    seq = root_empty.find("sequence")
    assert seq.find("duration").text == "0"
    assert len(seq.findall(".//media/video/track/clipitem")) == 0

    # Non-positive segments should be filtered out
    invalid_segments = [(5.0, 5.0), (8.0, 6.0), (1.0, 3.0)]
    xml_filtered = generate_fcp7_xml("/path/to/test.mp4", invalid_segments, fps=30.0)
    root_filtered = ET.fromstring(xml_filtered)
    clips = root_filtered.findall(".//media/video/track/clipitem")
    assert len(clips) == 1
    assert clips[0].find("in").text == "30"
    assert clips[0].find("out").text == "90"


def test_generate_fcp7_xml_ntsc():
    xml = generate_fcp7_xml("/path/to/ntsc.mp4", [(0.0, 5.0)], fps=29.97)
    root = ET.fromstring(xml)
    seq = root.find("sequence")
    assert seq.find("rate/timebase").text == "30"
    assert seq.find("rate/ntsc").text == "TRUE"


def test_generate_cmx3600_edl():
    segments = [(0.0, 4.0), (10.0, 14.0)]
    edl = generate_cmx3600_edl("/path/to/video.mp4", segments, fps=30.0, sequence_name="Interview_Cut")

    # Header check
    assert "TITLE: Interview_Cut" in edl
    assert "FCM: NON-DROP FRAME" in edl

    # Events check
    assert "001" in edl
    assert "002" in edl
    # Event 1: src 00:00:00:00 00:00:04:00, rec 00:00:00:00 00:00:04:00
    assert "00:00:00:00 00:00:04:00 00:00:00:00 00:00:04:00" in edl
    # Event 2: src 00:00:10:00 00:00:14:00, rec 00:00:04:00 00:00:08:00
    assert "00:00:10:00 00:00:14:00 00:00:04:00 00:00:08:00" in edl

    # Clip comment check
    assert "* FROM CLIP NAME: video.mp4" in edl


def test_generate_cmx3600_edl_empty():
    edl = generate_cmx3600_edl("/path/to/empty.mp4", [], fps=25.0)
    assert "TITLE: Interview_Cut" in edl
    assert "FCM: NON-DROP FRAME" in edl
    assert "001" not in edl


def test_export_timeline_files_io(tmp_path):
    output_base = str(tmp_path / "test_session")
    segments = [(0.0, 2.0), (5.0, 7.0)]
    source_media = "/path/to/interview.mp4"

    res = export_timeline_files(
        source_media_path=source_media,
        segments=segments,
        fps=30.0,
        output_base_path=output_base,
        width=1920,
        height=1080
    )

    expected_xml = f"{output_base}_timeline.xml"
    expected_edl = f"{output_base}_timeline.edl"

    assert res["xml"] == expected_xml
    assert res["edl"] == expected_edl

    assert os.path.exists(expected_xml)
    assert os.path.exists(expected_edl)

    # Verify XML is valid and non-empty
    tree = ET.parse(expected_xml)
    assert tree.getroot().tag == "xmeml"

    # Verify EDL is valid and non-empty
    with open(expected_edl, "r", encoding="utf-8") as f:
        edl_content = f.read()
    assert "TITLE: Interview_Cut" in edl_content
    assert "001" in edl_content
    assert "002" in edl_content
