#!/usr/bin/env python3
import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openpilot.tools.lib.logreader import LogReader


ROW_KEYS = [
  "mono_time_s",
  "enabled",
  "lateral_active",
  "state",
  "source",
  "map_valid",
  "vision_valid",
  "curvature_1_per_m",
  "distance_m",
  "time_to_bend_s",
  "required_lateral_accel_mps2",
  "safe_speed_kph",
  "current_speed_kph",
  "candidate_time_s",
  "episode",
  "rejection_reason",
  "event_active",
]

EVENT_STATES = {"warning", "clearing"}


def _enum_name(value: Any) -> str:
  return str(value)


def row_from_message(msg: Any) -> dict[str, Any]:
  warning = msg.longitudinalPlanSP.bendWarning
  state = _enum_name(warning.state)
  return {
    "mono_time_s": msg.logMonoTime / 1e9,
    "enabled": bool(warning.enabled),
    "lateral_active": bool(warning.lateralActive),
    "state": state,
    "source": _enum_name(warning.source),
    "map_valid": bool(warning.mapValid),
    "vision_valid": bool(warning.visionValid),
    "curvature_1_per_m": float(warning.curvature),
    "distance_m": float(warning.distance),
    "time_to_bend_s": float(warning.timeToBend),
    "required_lateral_accel_mps2": float(warning.requiredLateralAccel),
    "safe_speed_kph": float(warning.safeSpeed) * 3.6,
    "current_speed_kph": float(warning.currentSpeed) * 3.6,
    "candidate_time_s": float(warning.candidateTime),
    "episode": int(warning.episode),
    "rejection_reason": _enum_name(warning.rejectionReason),
    "event_active": state in EVENT_STATES,
  }


def analyze_messages(messages: Iterable[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  rows = [row_from_message(msg) for msg in messages if msg.which() == "longitudinalPlanSP"]

  episode_sources: dict[int, str] = {}
  minimum_ttb: dict[int, float] = {}
  entry_counts: dict[int, int] = {}
  episode_starts = 0
  episode_ends = 0
  event_entries = 0
  previous_active = False

  for row in rows:
    active = row["event_active"]
    episode = row["episode"]
    if active and not previous_active:
      episode_starts += 1
      event_entries += 1
      entry_counts[episode] = entry_counts.get(episode, 0) + 1
      episode_sources.setdefault(episode, row["source"])
    elif previous_active and not active:
      episode_ends += 1

    if active and row["time_to_bend_s"] > 0:
      minimum_ttb[episode] = min(minimum_ttb.get(episode, float("inf")), row["time_to_bend_s"])
    previous_active = active

  first_warning = next((row["mono_time_s"] for row in rows if row["event_active"]), None)
  summary = {
    "total_samples": len(rows),
    "warning_episodes": len(episode_sources),
    "episode_starts": episode_starts,
    "episode_ends": episode_ends,
    "event_entries": event_entries,
    "map_episodes": sum(source == "map" for source in episode_sources.values()),
    "vision_episodes": sum(source == "vision" for source in episode_sources.values()),
    "both_episodes": sum(source == "both" for source in episode_sources.values()),
    "earliest_warning_time_s": first_warning,
    "minimum_time_to_bend_by_episode_s": {str(episode): minimum_ttb[episode] for episode in sorted(minimum_ttb)},
    "warnings_below_50_kph": sum(row["event_active"] and row["current_speed_kph"] < 50.0 for row in rows),
    "warnings_lateral_inactive": sum(row["event_active"] and not row["lateral_active"] for row in rows),
    "repeated_episode_sounds": sum(max(entries - 1, 0) for entries in entry_counts.values()),
  }
  return rows, summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Dump predictive bend warning diagnostics from a replay log")
  parser.add_argument("input_rlog", help="Input rlog or process-replay log")
  parser.add_argument("--jsonl", required=True, help="Output JSONL path")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  rows, summary = analyze_messages(LogReader(args.input_rlog))
  output_path = Path(args.jsonl)
  with output_path.open("w") as output:
    for row in rows:
      output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

  print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
  unsafe = (
    summary["warnings_below_50_kph"] > 0
    or summary["warnings_lateral_inactive"] > 0
    or summary["repeated_episode_sounds"] > 0
  )
  return int(unsafe)


if __name__ == "__main__":
  raise SystemExit(main())
