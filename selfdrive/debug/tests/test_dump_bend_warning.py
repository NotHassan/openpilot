import json
import subprocess
import sys

import pytest

from cereal import messaging
from openpilot.selfdrive.debug.dump_bend_warning import ROW_KEYS, analyze_messages, main
from openpilot.selfdrive.debug import run_process_on_route
from openpilot.selfdrive.debug.run_process_on_route import build_custom_params, parse_param_assignments
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config
from openpilot.tools.lib.logreader import LogReader, save_log


def bend_warning_message(
  mono_time_s: float,
  *,
  state: str = "idle",
  source: str = "none",
  episode: int = 0,
  current_speed_kph: float = 108.0,
  lateral_active: bool = True,
  rejection_reason: str = "none",
  map_valid: bool = True,
  vision_valid: bool = False,
  time_to_bend_s: float = 6.0,
  serialized_event: bool | None = None,
  event_name: str = "predictiveBendWarning",
  event_warning: bool = True,
):
  msg = messaging.new_message("longitudinalPlanSP")
  msg.logMonoTime = int(mono_time_s * 1e9)
  plan = msg.longitudinalPlanSP
  warning = plan.bendWarning
  warning.enabled = True
  warning.lateralActive = lateral_active
  warning.state = state
  warning.source = source
  warning.mapValid = map_valid
  warning.visionValid = vision_valid
  warning.curvature = 0.003
  warning.distance = 180.0
  warning.timeToBend = time_to_bend_s
  warning.requiredLateralAccel = 2.7
  warning.safeSpeed = 93.0 / 3.6
  warning.currentSpeed = current_speed_kph / 3.6
  warning.candidateTime = 0.5
  warning.episode = episode
  warning.rejectionReason = rejection_reason
  if serialized_event is None:
    serialized_event = state in {"warning", "clearing"}
  if serialized_event:
    [event] = plan.init("events", 1)
    event.name = event_name
    event.warning = event_warning
  return msg.as_reader()


def test_stable_jsonl_rows_and_episode_summary(tmp_path):
  input_log = tmp_path / "synthetic.zst"
  messages = [
    bend_warning_message(12.0, state="candidate", source="map", episode=0),
    bend_warning_message(12.5, state="warning", source="map", episode=1, time_to_bend_s=6.0),
    bend_warning_message(13.0, state="clearing", source="map", episode=1, time_to_bend_s=5.5),
    bend_warning_message(16.0, state="idle", source="none", episode=1, map_valid=False,
                         rejection_reason="staleSegment"),
    bend_warning_message(20.0, state="warning", source="both", episode=2, time_to_bend_s=4.0),
    bend_warning_message(21.0, state="idle", source="none", episode=2),
  ]
  save_log(str(input_log), messages)

  rows, summary = analyze_messages(LogReader(str(input_log)))

  assert len(rows) == len(messages)
  assert list(rows[0]) == ROW_KEYS
  assert rows[1] == {
    "mono_time_s": 12.5,
    "enabled": True,
    "lateral_active": True,
    "state": "warning",
    "source": "map",
    "map_valid": True,
    "vision_valid": False,
    "curvature_1_per_m": pytest.approx(0.003),
    "distance_m": pytest.approx(180.0),
    "time_to_bend_s": pytest.approx(6.0),
    "required_lateral_accel_mps2": pytest.approx(2.7),
    "safe_speed_kph": pytest.approx(93.0),
    "current_speed_kph": pytest.approx(108.0),
    "candidate_time_s": pytest.approx(0.5),
    "episode": 1,
    "rejection_reason": "none",
    "event_active": True,
  }
  assert rows[3]["rejection_reason"] == "staleSegment"
  assert not rows[3]["map_valid"]
  assert summary == {
    "total_samples": 6,
    "warning_episodes": 2,
    "episode_starts": 2,
    "episode_ends": 2,
    "event_entries": 2,
    "map_episodes": 1,
    "vision_episodes": 0,
    "both_episodes": 1,
    "earliest_warning_time_s": 12.5,
    "minimum_time_to_bend_by_episode_s": {"1": 5.5, "2": 4.0},
    "warnings_below_50_kph": 0,
    "warnings_lateral_inactive": 0,
    "repeated_episode_sounds": 0,
  }


def test_invalid_preview_sample_is_not_dropped():
  msg = bend_warning_message(2.0, state="idle", source="map", map_valid=False,
                             rejection_reason="ambiguousPath", time_to_bend_s=0.0)
  rows, summary = analyze_messages([msg])
  assert len(rows) == 1
  assert rows[0]["source"] == "map"
  assert rows[0]["rejection_reason"] == "ambiguousPath"
  assert summary["total_samples"] == 1


@pytest.mark.parametrize(
  "message, expected_active",
  [
    (bend_warning_message(1.0, state="warning", source="map", episode=1, serialized_event=False), False),
    (bend_warning_message(1.0, state="idle", source="map", episode=1, serialized_event=True), True),
    (bend_warning_message(1.0, state="warning", source="map", episode=1,
                          serialized_event=True, event_warning=False), False),
    (bend_warning_message(1.0, state="warning", source="map", episode=1,
                          serialized_event=True, event_name="speedLimitActive"), False),
  ],
)
def test_event_active_comes_from_serialized_warning_event(message, expected_active):
  rows, _ = analyze_messages([message])
  assert rows[0]["event_active"] is expected_active


def test_repeated_serialized_event_entry_in_one_episode_is_counted():
  messages = [
    bend_warning_message(1.0, state="warning", source="map", episode=1, serialized_event=True),
    bend_warning_message(2.0, state="warning", source="map", episode=1, serialized_event=False),
    bend_warning_message(3.0, state="clearing", source="map", episode=1, serialized_event=True),
  ]

  _, summary = analyze_messages(messages)

  assert summary["event_entries"] == 2
  assert summary["repeated_episode_sounds"] == 1


def test_safety_counts_follow_serialized_event_not_diagnostic_state():
  messages = [
    bend_warning_message(1.0, state="warning", episode=1, current_speed_kph=49.0,
                         lateral_active=False, serialized_event=False),
    bend_warning_message(2.0, state="idle", episode=2, current_speed_kph=49.0,
                         lateral_active=False, serialized_event=True),
  ]

  _, summary = analyze_messages(messages)

  assert summary["warnings_below_50_kph"] == 1
  assert summary["warnings_lateral_inactive"] == 1


@pytest.mark.parametrize(
  "messages, expected_field",
  [
    ([bend_warning_message(1.0, state="warning", source="map", episode=1, current_speed_kph=49.9)],
     "warnings_below_50_kph"),
    ([bend_warning_message(1.0, state="warning", source="vision", episode=1, lateral_active=False)],
     "warnings_lateral_inactive"),
    ([
      bend_warning_message(1.0, state="warning", source="map", episode=1),
      bend_warning_message(2.0, state="idle", source="none", episode=1),
      bend_warning_message(3.0, state="warning", source="map", episode=1),
    ], "repeated_episode_sounds"),
  ],
)
def test_safety_violations_return_nonzero(tmp_path, messages, expected_field):
  input_log = tmp_path / "unsafe.zst"
  output_jsonl = tmp_path / "unsafe.jsonl"
  save_log(str(input_log), messages)

  assert main([str(input_log), "--jsonl", str(output_jsonl)]) == 1
  _, summary = analyze_messages(messages)
  assert summary[expected_field] > 0
  assert len(output_jsonl.read_text().splitlines()) == len(messages)


def test_cli_writes_stable_jsonl_and_prints_summary(tmp_path):
  input_log = tmp_path / "safe.zst"
  output_jsonl = tmp_path / "safe.jsonl"
  save_log(str(input_log), [
    bend_warning_message(1.0, state="warning", source="vision", episode=1),
    bend_warning_message(2.0, state="idle", source="none", episode=1),
  ])

  result = subprocess.run(
    [sys.executable, "selfdrive/debug/dump_bend_warning.py", str(input_log), "--jsonl", str(output_jsonl)],
    check=False, capture_output=True, text=True,
  )

  assert result.returncode == 0
  assert json.loads(result.stdout)["warning_episodes"] == 1
  assert [json.loads(line)["mono_time_s"] for line in output_jsonl.read_text().splitlines()] == [1.0, 2.0]


def test_parse_param_assignments():
  assert parse_param_assignments(["PredictiveBendWarning=1", "Empty="]) == {
    "PredictiveBendWarning": b"1",
    "Empty": b"",
  }


def test_replay_override_merge_and_captured_output(monkeypatch):
  monkeypatch.setattr(run_process_on_route, "get_custom_params_from_lr",
                      lambda inputs: {"CarParamsPrevRoute": b"recorded", "PredictiveBendWarning": b"0"})

  custom_params = build_custom_params([object()], ["PredictiveBendWarning=1"])

  assert custom_params == {"CarParamsPrevRoute": b"recorded", "PredictiveBendWarning": b"1"}
  assert "longitudinalPlanSP" in get_process_config("plannerd").subs


@pytest.mark.parametrize("assignment", ["PredictiveBendWarning", "=1"])
def test_parse_param_assignments_rejects_malformed_values(assignment):
  with pytest.raises(ValueError, match=r"expected KEY=VALUE"):
    parse_param_assignments([assignment])
