from vision.ensemble import Detection, box_iou, fuse_detections
from vision.tracking import IoUTracker


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_fusion_averages_boxes_and_counts_independent_models():
    detections = [
        Detection((0, 0, 100, 100), "car", 0.8, {"small.pt"}),
        Detection((4, 2, 104, 102), "car", 0.9, {"medium.pt"}),
        Detection((200, 200, 250, 250), "person", 0.9, {"small.pt"}),
    ]
    fused = fuse_detections(detections, iou_threshold=0.5, min_model_votes=2)
    assert len(fused) == 1
    assert fused[0].label == "car"
    assert fused[0].model_votes == 2
    assert 0 < fused[0].box[0] < 4


def test_tracker_preserves_identity_and_computes_ttc():
    tracker = IoUTracker(minimum_iou=0.1, max_age_seconds=2.0)
    first = Detection((200, 100, 300, 300), "car", 0.9, {"a"})
    tracker.update([first], timestamp=0.0, frame_width=640, frame_height=480)

    second = Detection((150, 50, 350, 350), "car", 0.9, {"a"})
    tracker.update([second], timestamp=1.0, frame_width=640, frame_height=480)

    assert second.track_id == first.track_id
    assert second.ttc_status == "closing_in"
    assert round(second.ttc_seconds, 3) == 2.0


def test_tracker_measures_normalized_lateral_motion():
    tracker = IoUTracker(minimum_iou=0.05, max_age_seconds=2.0)
    first = Detection((0, 100, 100, 200), "motorcycle", 0.9, {"a"})
    tracker.update([first], timestamp=0.0, frame_width=1000, frame_height=500)

    second = Detection((50, 100, 150, 200), "motorcycle", 0.9, {"a"})
    tracker.update([second], timestamp=1.0, frame_width=1000, frame_height=500)
    assert second.track_id == first.track_id
    assert second.lateral_velocity == 0.05


def test_tracker_treats_stable_close_object_as_no_collision():
    tracker = IoUTracker(minimum_iou=0.1, max_age_seconds=5.0)
    first = Detection((100, 100, 500, 460), "bus", 0.9, {"a"})
    tracker.update([first], timestamp=0.0, frame_width=640, frame_height=480)

    second = Detection((100, 100, 500, 460), "bus", 0.9, {"a"})
    tracker.update([second], timestamp=3.0, frame_width=640, frame_height=480)

    assert second.ttc_status == "stable"
    assert second.ttc_seconds == float("inf")
    assert second.stable_seconds == 3.0
