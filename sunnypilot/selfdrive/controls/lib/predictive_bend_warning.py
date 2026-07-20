import math
from dataclasses import dataclass
from enum import IntEnum

from opendbc.car.common.conversions import Conversions as CV

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.models.constants import ModelConstants


ENTER_SPEED = 50.0 * CV.KPH_TO_MS
EXIT_SPEED = 45.0 * CV.KPH_TO_MS
WARNING_LAT_ACCEL = 2.0
MIN_SAFE_SPEED_DELTA = 5.0 * CV.KPH_TO_MS
MAX_TIME_TO_BEND = 8.0
PERSISTENCE_FRAMES = round(0.5 / DT_MDL)
CLEAR_FRAMES = round(3.0 / DT_MDL)
MIN_MODEL_SPEED = 1.0


class WarningState(IntEnum):
  idle = 0
  candidate = 1
  warning = 2
  clearing = 3


class WarningSource(IntEnum):
  none = 0
  map = 1
  vision = 2
  both = 3


class RejectionReason(IntEnum):
  none = 0
  disabled = 1
  lateralInactive = 2
  belowSpeed = 3
  sourceUnavailable = 4
  ambiguousLocation = 5
  locationError = 6
  ambiguousPath = 7
  staleSegment = 8
  invalidCurvature = 9
  invalidDistance = 10
  sanityFilter = 11


@dataclass(frozen=True)
class BendPrediction:
  valid: bool = False
  unsafe: bool = False
  curvature: float = 0.0
  distance: float = 0.0
  time_to_bend: float = 0.0
  required_lateral_accel: float = 0.0
  safe_speed: float = 0.0
  rejection_reason: RejectionReason = RejectionReason.sourceUnavailable


@dataclass(frozen=True)
class BendWarningOutput:
  enabled: bool = False
  lateral_active: bool = False
  state: WarningState = WarningState.idle
  source: WarningSource = WarningSource.none
  map_valid: bool = False
  vision_valid: bool = False
  curvature: float = 0.0
  distance: float = 0.0
  time_to_bend: float = 0.0
  required_lateral_accel: float = 0.0
  safe_speed: float = 0.0
  current_speed: float = 0.0
  candidate_time: float = 0.0
  episode: int = 0
  rejection_reason: RejectionReason = RejectionReason.sourceUnavailable
  emit_event: bool = False
  map_prediction: BendPrediction = BendPrediction()
  vision_prediction: BendPrediction = BendPrediction()


class PredictiveBendWarning:
  def __init__(self, enabled: bool):
    self.enabled = enabled
    self.state = WarningState.idle
    self.speed_eligible = False
    self.candidate_frames = 0
    self.clear_frames = 0
    self.episode = 0

  @staticmethod
  def _is_unsafe(v_ego: float, curvature: float, safe_speed: float) -> bool:
    required_lateral_accel = v_ego ** 2 * abs(curvature)
    return required_lateral_accel >= WARNING_LAT_ACCEL and v_ego - safe_speed >= MIN_SAFE_SPEED_DELTA

  @staticmethod
  def _prediction(v_ego: float, curvature: float, distance: float, time_to_bend: float) -> BendPrediction:
    safe_speed = math.sqrt(WARNING_LAT_ACCEL / abs(curvature)) if curvature != 0.0 else math.inf
    return BendPrediction(
      valid=True,
      unsafe=PredictiveBendWarning._is_unsafe(v_ego, curvature, safe_speed),
      curvature=curvature,
      distance=distance,
      time_to_bend=time_to_bend,
      required_lateral_accel=v_ego ** 2 * abs(curvature),
      safe_speed=safe_speed,
      rejection_reason=RejectionReason.none,
    )

  @staticmethod
  def _map_rejection_reason(value) -> RejectionReason:
    if hasattr(value, "name"):
      name = value.name
    elif isinstance(value, str):
      name = value
    else:
      source_reason_ordinals = {
        0: RejectionReason.none,
        1: RejectionReason.sourceUnavailable,
        2: RejectionReason.ambiguousLocation,
        3: RejectionReason.locationError,
        4: RejectionReason.ambiguousPath,
        5: RejectionReason.staleSegment,
        6: RejectionReason.invalidCurvature,
        7: RejectionReason.invalidDistance,
        8: RejectionReason.sanityFilter,
      }
      return source_reason_ordinals.get(value, RejectionReason.sourceUnavailable)
    return RejectionReason.__members__.get(name, RejectionReason.sourceUnavailable)

  @classmethod
  def _evaluate_map(cls, v_ego: float, preview) -> BendPrediction:
    if preview is None or not bool(getattr(preview, "valid", False)):
      reason = cls._map_rejection_reason(getattr(preview, "rejectionReason", "sourceUnavailable"))
      return BendPrediction(rejection_reason=reason)

    curvature = float(getattr(preview, "curvature", math.nan))
    distance = float(getattr(preview, "distance", math.nan))
    if not math.isfinite(curvature) or curvature == 0.0:
      return BendPrediction(rejection_reason=RejectionReason.invalidCurvature)
    if not math.isfinite(distance) or distance < 0.0:
      return BendPrediction(rejection_reason=RejectionReason.invalidDistance)

    time_to_bend = distance / max(v_ego, MIN_MODEL_SPEED)
    return cls._prediction(v_ego, curvature, distance, time_to_bend)

  @classmethod
  def _evaluate_vision(cls, v_ego: float, model_data) -> BendPrediction:
    if model_data is None:
      return BendPrediction()

    orientation_rates = getattr(getattr(model_data, "orientationRate", None), "z", ())
    model_velocities = getattr(getattr(model_data, "velocity", None), "x", ())
    distances = getattr(getattr(model_data, "position", None), "x", ())
    if len(orientation_rates) == 0 or len(model_velocities) == 0 or len(distances) == 0:
      return BendPrediction()

    first_valid = None
    rejection_reason = RejectionReason.sourceUnavailable
    for orientation_rate, model_velocity, distance, time_to_bend in zip(
      orientation_rates, model_velocities, distances, ModelConstants.T_IDXS, strict=False,
    ):
      if time_to_bend <= 0.0:
        continue
      if not math.isfinite(orientation_rate):
        rejection_reason = RejectionReason.invalidCurvature
        continue
      if not math.isfinite(distance) or distance <= 0.0:
        if rejection_reason == RejectionReason.sourceUnavailable:
          rejection_reason = RejectionReason.invalidDistance
        continue
      if not math.isfinite(model_velocity) or model_velocity <= MIN_MODEL_SPEED:
        if rejection_reason == RejectionReason.sourceUnavailable:
          rejection_reason = RejectionReason.sanityFilter
        continue

      curvature = orientation_rate / model_velocity
      if not math.isfinite(curvature):
        rejection_reason = RejectionReason.invalidCurvature
        continue

      prediction = cls._prediction(v_ego, curvature, distance, time_to_bend)
      if first_valid is None:
        first_valid = prediction
      if prediction.unsafe:
        return prediction

    return first_valid or BendPrediction(rejection_reason=rejection_reason)

  def _reset_state(self, reset_speed_gate: bool = False) -> None:
    self.state = WarningState.idle
    self.candidate_frames = 0
    self.clear_frames = 0
    if reset_speed_gate:
      self.speed_eligible = False

  def _gated_output(self, lateral_active: bool, v_ego: float, reason: RejectionReason) -> BendWarningOutput:
    return BendWarningOutput(
      enabled=self.enabled,
      lateral_active=lateral_active,
      current_speed=v_ego,
      episode=self.episode,
      rejection_reason=reason,
    )

  @staticmethod
  def _fuse(map_prediction: BendPrediction, vision_prediction: BendPrediction):
    map_unsafe = map_prediction.valid and map_prediction.unsafe
    vision_unsafe = vision_prediction.valid and vision_prediction.unsafe
    if map_unsafe and vision_unsafe:
      return WarningSource.both, min(
        (map_prediction, vision_prediction),
        key=lambda prediction: prediction.time_to_bend,
      )
    if map_unsafe:
      return WarningSource.map, map_prediction
    if vision_unsafe:
      return WarningSource.vision, vision_prediction
    return WarningSource.none, None

  @staticmethod
  def _rejection_reason(map_prediction: BendPrediction, vision_prediction: BendPrediction) -> RejectionReason:
    if map_prediction.valid or vision_prediction.valid:
      return RejectionReason.none
    if map_prediction.rejection_reason != RejectionReason.sourceUnavailable:
      return map_prediction.rejection_reason
    return vision_prediction.rejection_reason

  def update(self, lateral_active, v_ego, map_preview, model_data) -> BendWarningOutput:
    if not self.enabled:
      self._reset_state(reset_speed_gate=True)
      return self._gated_output(lateral_active, v_ego, RejectionReason.disabled)
    if not lateral_active:
      self._reset_state(reset_speed_gate=True)
      return self._gated_output(lateral_active, v_ego, RejectionReason.lateralInactive)
    if v_ego < EXIT_SPEED:
      self._reset_state(reset_speed_gate=True)
      return self._gated_output(lateral_active, v_ego, RejectionReason.belowSpeed)

    if v_ego >= ENTER_SPEED:
      self.speed_eligible = True
    if not self.speed_eligible:
      self._reset_state()
      return self._gated_output(lateral_active, v_ego, RejectionReason.belowSpeed)

    map_prediction = self._evaluate_map(v_ego, map_preview)
    vision_prediction = self._evaluate_vision(v_ego, model_data)
    source, selected = self._fuse(map_prediction, vision_prediction)
    unsafe = selected is not None
    in_window_risk = unsafe and 0.0 < selected.time_to_bend <= MAX_TIME_TO_BEND

    if self.state == WarningState.idle:
      if in_window_risk:
        self.state = WarningState.candidate
        self.candidate_frames = 1
    elif self.state == WarningState.candidate:
      if not in_window_risk:
        self._reset_state()
      else:
        self.candidate_frames += 1
        if self.candidate_frames >= PERSISTENCE_FRAMES:
          self.state = WarningState.warning
          self.episode += 1
    elif self.state == WarningState.warning:
      if not in_window_risk:
        self.state = WarningState.clearing
        self.clear_frames = 1
    elif self.state == WarningState.clearing:
      if in_window_risk:
        self.state = WarningState.warning
        self.clear_frames = 0
      else:
        self.clear_frames += 1
        if self.clear_frames >= CLEAR_FRAMES:
          self._reset_state()

    selected = selected or BendPrediction()
    return BendWarningOutput(
      enabled=self.enabled,
      lateral_active=lateral_active,
      state=self.state,
      source=source,
      map_valid=map_prediction.valid,
      vision_valid=vision_prediction.valid,
      curvature=selected.curvature,
      distance=selected.distance,
      time_to_bend=selected.time_to_bend,
      required_lateral_accel=selected.required_lateral_accel,
      safe_speed=selected.safe_speed,
      current_speed=v_ego,
      candidate_time=min(self.candidate_frames, PERSISTENCE_FRAMES) * DT_MDL,
      episode=self.episode,
      rejection_reason=self._rejection_reason(map_prediction, vision_prediction),
      emit_event=self.state in (WarningState.warning, WarningState.clearing),
      map_prediction=map_prediction,
      vision_prediction=vision_prediction,
    )
