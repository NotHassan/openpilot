import inspect
import math
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.models.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.predictive_bend_warning import (
  CLEAR_FRAMES,
  ENTER_SPEED,
  EXIT_SPEED,
  PERSISTENCE_FRAMES,
  BendPrediction,
  BendWarningOutput,
  PredictiveBendWarning,
  RejectionReason,
  WarningSource,
  WarningState,
)


V_EGO = 30.0
UNSAFE_CURVATURE = 0.004
SAFE_CURVATURE = 0.001


def test_parameter_is_registered_default_off():
  params = Params()
  params.remove("PredictiveBendWarning")

  assert not params.get_bool("PredictiveBendWarning")
  assert params.get("PredictiveBendWarning", return_default=True) is False


def map_preview(curvature=UNSAFE_CURVATURE, distance=150.0, *, valid=True, rejection_reason="none"):
  return SimpleNamespace(
    valid=valid,
    curvature=curvature,
    distance=distance,
    rejectionReason=rejection_reason,
  )


def empty_model():
  return SimpleNamespace(
    orientationRate=SimpleNamespace(z=[]),
    velocity=SimpleNamespace(x=[]),
    position=SimpleNamespace(x=[]),
  )


def model_with_points(points):
  """Build model arrays where points maps a model index to (curvature, distance, velocity)."""
  size = max(points, default=0) + 1
  orientation_rate = [math.nan] * size
  velocity = [math.nan] * size
  position = [math.nan] * size
  for index, (curvature, distance, model_velocity) in points.items():
    orientation_rate[index] = curvature * model_velocity
    velocity[index] = model_velocity
    position[index] = distance
  return SimpleNamespace(
    orientationRate=SimpleNamespace(z=orientation_rate),
    velocity=SimpleNamespace(x=velocity),
    position=SimpleNamespace(x=position),
  )


def first_index_at_or_after(seconds):
  return next(index for index, value in enumerate(ModelConstants.T_IDXS) if value >= seconds)


def advance_to_warning(controller, preview=None, model=None):
  preview = preview or map_preview()
  model = model or empty_model()
  output = None
  for _ in range(PERSISTENCE_FRAMES):
    output = controller.update(True, V_EGO, preview, model)
  assert output is not None
  return output


def test_contract_constants_enum_ordinals_and_immutable_outputs():
  assert ENTER_SPEED == pytest.approx(50.0 / 3.6)
  assert EXIT_SPEED == pytest.approx(45.0 / 3.6)
  assert PERSISTENCE_FRAMES == round(0.5 / DT_MDL)
  assert CLEAR_FRAMES == round(3.0 / DT_MDL)
  assert [state.value for state in WarningState] == [0, 1, 2, 3]
  assert [source.value for source in WarningSource] == [0, 1, 2, 3]
  assert [reason.value for reason in RejectionReason] == list(range(12))

  prediction = BendPrediction()
  output = BendWarningOutput()
  with pytest.raises(FrozenInstanceError):
    prediction.valid = True
  with pytest.raises(FrozenInstanceError):
    output.emit_event = True


def test_controller_interface_is_advisory_only():
  assert list(inspect.signature(PredictiveBendWarning.update).parameters) == [
    "self", "lateral_active", "v_ego", "map_preview", "model_data",
  ]
  forbidden = ("cruise", "target", "accel_command", "steer_command", "button")
  output_fields = {field.name for field in fields(BendWarningOutput)}
  assert all(not any(token in name for token in forbidden) for name in output_fields)


def test_map_calculations_use_current_vehicle_speed():
  output = PredictiveBendWarning(enabled=True).update(
    True, V_EGO, map_preview(curvature=-UNSAFE_CURVATURE, distance=120.0), empty_model(),
  )

  prediction = output.map_prediction
  assert prediction.curvature == -UNSAFE_CURVATURE
  assert prediction.required_lateral_accel == pytest.approx(V_EGO ** 2 * UNSAFE_CURVATURE)
  assert prediction.safe_speed == pytest.approx(math.sqrt(2.0 / UNSAFE_CURVATURE))
  assert prediction.time_to_bend == pytest.approx(120.0 / V_EGO)


def test_camera_calculates_curvature_and_selects_earliest_unsafe_point():
  early_index = first_index_at_or_after(2.0)
  later_index = first_index_at_or_after(4.0)
  model = model_with_points({
    early_index: (-UNSAFE_CURVATURE, 65.0, 25.0),
    later_index: (0.012, 125.0, 25.0),
  })

  output = PredictiveBendWarning(enabled=True).update(True, V_EGO, map_preview(valid=False), model)

  prediction = output.vision_prediction
  assert prediction.curvature == pytest.approx(-UNSAFE_CURVATURE)
  assert prediction.distance == 65.0
  assert prediction.time_to_bend == ModelConstants.T_IDXS[early_index]
  assert prediction.required_lateral_accel == pytest.approx(V_EGO ** 2 * UNSAFE_CURVATURE)
  assert prediction.safe_speed == pytest.approx(math.sqrt(2.0 / UNSAFE_CURVATURE))


@pytest.mark.parametrize(
  ("preview", "model", "expected_source", "map_valid", "vision_valid"),
  [
    (map_preview(), empty_model(), WarningSource.map, True, False),
    (
      map_preview(valid=False),
      model_with_points({first_index_at_or_after(3.0): (UNSAFE_CURVATURE, 90.0, 30.0)}),
      WarningSource.vision,
      False,
      True,
    ),
    (
      map_preview(distance=90.0),
      model_with_points({first_index_at_or_after(3.0): (UNSAFE_CURVATURE, 90.0, 30.0)}),
      WarningSource.both,
      True,
      True,
    ),
  ],
)
def test_map_only_vision_only_and_agreeing_sources(preview, model, expected_source, map_valid, vision_valid):
  output = PredictiveBendWarning(enabled=True).update(True, V_EGO, preview, model)

  assert output.state == WarningState.candidate
  assert output.source == expected_source
  assert output.map_valid is map_valid
  assert output.vision_valid is vision_valid


def test_disagreeing_unsafe_sources_choose_earliest_and_log_both():
  camera_index = first_index_at_or_after(3.0)
  camera_curvature = 0.006
  output = PredictiveBendWarning(enabled=True).update(
    True,
    V_EGO,
    map_preview(curvature=UNSAFE_CURVATURE, distance=180.0),
    model_with_points({camera_index: (camera_curvature, 90.0, 30.0)}),
  )

  assert output.source == WarningSource.both
  assert output.curvature == pytest.approx(camera_curvature)
  assert output.distance == 90.0
  assert output.time_to_bend == ModelConstants.T_IDXS[camera_index]


def test_speed_gate_has_50_45_kph_hysteresis_and_does_not_arm_below_entry():
  controller = PredictiveBendWarning(enabled=True)
  low_speed_unsafe_preview = map_preview(curvature=0.02, distance=50.0)

  output = controller.update(True, 47.0 / 3.6, low_speed_unsafe_preview, empty_model())
  assert output.state == WarningState.idle
  assert output.rejection_reason == RejectionReason.belowSpeed

  output = controller.update(True, 50.0 / 3.6, low_speed_unsafe_preview, empty_model())
  assert output.state == WarningState.candidate

  output = controller.update(True, 47.0 / 3.6, low_speed_unsafe_preview, empty_model())
  assert output.state == WarningState.candidate

  output = controller.update(True, 44.9 / 3.6, low_speed_unsafe_preview, empty_model())
  assert output.state == WarningState.idle
  assert output.rejection_reason == RejectionReason.belowSpeed


def test_persistent_map_candidate_warns_on_exact_half_second_and_once():
  controller = PredictiveBendWarning(enabled=True)
  for frame in range(1, PERSISTENCE_FRAMES):
    output = controller.update(True, V_EGO, map_preview(), empty_model())
    assert output.state == WarningState.candidate
    assert output.candidate_time == pytest.approx(frame * DT_MDL)
    assert not output.emit_event

  output = controller.update(True, V_EGO, map_preview(), empty_model())
  assert output.state == WarningState.warning
  assert output.candidate_time == pytest.approx(0.5)
  assert output.emit_event
  assert output.episode == 1

  output = controller.update(True, V_EGO, map_preview(), empty_model())
  assert output.state == WarningState.warning
  assert output.emit_event
  assert output.episode == 1


def test_tracks_unsafe_bend_silently_above_eight_seconds_then_forms_candidate_at_boundary():
  controller = PredictiveBendWarning(enabled=True)

  output = controller.update(True, V_EGO, map_preview(distance=V_EGO * 8.01), empty_model())
  assert output.state == WarningState.idle
  assert output.source == WarningSource.map
  assert output.map_prediction.unsafe
  assert not output.emit_event

  output = controller.update(True, V_EGO, map_preview(distance=V_EGO * 8.0), empty_model())
  assert output.state == WarningState.candidate
  assert output.candidate_time == pytest.approx(DT_MDL)


def test_first_credible_point_below_five_seconds_forms_candidate_immediately_but_still_persists():
  controller = PredictiveBendWarning(enabled=True)
  controller.update(True, V_EGO, map_preview(distance=V_EGO * 9.0), empty_model())

  output = controller.update(True, V_EGO, map_preview(distance=V_EGO * 4.0), empty_model())
  assert output.state == WarningState.candidate
  assert output.candidate_time == pytest.approx(DT_MDL)
  assert not output.emit_event

  for _ in range(PERSISTENCE_FRAMES - 1):
    output = controller.update(True, V_EGO, map_preview(distance=V_EGO * 4.0), empty_model())
  assert output.state == WarningState.warning
  assert output.emit_event


def test_source_dropout_enters_clearing_and_rearms_after_exactly_three_seconds():
  controller = PredictiveBendWarning(enabled=True)
  output = advance_to_warning(controller)
  assert output.episode == 1

  for _ in range(1, CLEAR_FRAMES):
    output = controller.update(True, V_EGO, map_preview(valid=False), empty_model())
    assert output.state == WarningState.clearing
    assert output.emit_event
    assert output.episode == 1

  output = controller.update(True, V_EGO, map_preview(valid=False), empty_model())
  assert output.state == WarningState.idle
  assert not output.emit_event
  assert output.episode == 1

  output = advance_to_warning(controller)
  assert output.state == WarningState.warning
  assert output.episode == 2


def test_out_of_window_unsafe_prediction_during_warning_enters_clearing():
  controller = PredictiveBendWarning(enabled=True)
  advance_to_warning(controller)

  output = controller.update(
    True, V_EGO, map_preview(distance=V_EGO * 9.0), empty_model(),
  )

  assert output.map_prediction.unsafe
  assert output.time_to_bend == pytest.approx(9.0)
  assert output.state == WarningState.clearing
  assert output.emit_event


def test_out_of_window_unsafe_prediction_during_clearing_does_not_revive_warning():
  controller = PredictiveBendWarning(enabled=True)
  advance_to_warning(controller)
  output = controller.update(True, V_EGO, map_preview(valid=False), empty_model())
  assert output.state == WarningState.clearing

  output = controller.update(
    True, V_EGO, map_preview(distance=V_EGO * 9.0), empty_model(),
  )

  assert output.map_prediction.unsafe
  assert output.state == WarningState.clearing
  assert output.emit_event
  assert output.episode == 1


def test_out_of_window_risk_holds_event_only_for_normal_clearing_grace_then_rearms():
  controller = PredictiveBendWarning(enabled=True)
  assert advance_to_warning(controller).episode == 1

  for _ in range(CLEAR_FRAMES - 1):
    output = controller.update(
      True, V_EGO, map_preview(distance=V_EGO * 9.0), empty_model(),
    )
    assert output.state == WarningState.clearing
    assert output.emit_event
    assert output.episode == 1

  output = controller.update(
    True, V_EGO, map_preview(distance=V_EGO * 9.0), empty_model(),
  )
  assert output.state == WarningState.idle
  assert not output.emit_event
  assert output.episode == 1
  assert output.map_prediction.unsafe

  output = advance_to_warning(controller, map_preview(distance=V_EGO * 4.0))
  assert output.state == WarningState.warning
  assert output.emit_event
  assert output.episode == 2


def test_connected_bends_stay_in_one_episode_without_a_second_entry():
  controller = PredictiveBendWarning(enabled=True)
  assert advance_to_warning(controller).episode == 1

  for _ in range(CLEAR_FRAMES - 1):
    output = controller.update(
      True, V_EGO, map_preview(curvature=SAFE_CURVATURE), empty_model(),
    )
    assert output.state == WarningState.clearing
    assert output.emit_event

  output = controller.update(True, V_EGO, map_preview(distance=80.0), empty_model())
  assert output.state == WarningState.warning
  assert output.emit_event
  assert output.episode == 1


@pytest.mark.parametrize(
  ("enabled", "lateral_active", "expected_reason"),
  [
    (False, True, RejectionReason.disabled),
    (True, False, RejectionReason.lateralInactive),
  ],
)
def test_disabled_or_lateral_inactive_resets_immediately(enabled, lateral_active, expected_reason):
  controller = PredictiveBendWarning(enabled=True)
  assert advance_to_warning(controller).emit_event
  controller.enabled = enabled

  output = controller.update(lateral_active, V_EGO, map_preview(), empty_model())
  assert output.state == WarningState.idle
  assert not output.emit_event
  assert output.rejection_reason == expected_reason


def test_safe_speed_deficit_under_five_kph_never_warns():
  controller = PredictiveBendWarning(enabled=True)
  curvature = 2.0 / ((V_EGO - (4.9 / 3.6)) ** 2)

  for _ in range(PERSISTENCE_FRAMES + 1):
    output = controller.update(True, V_EGO, map_preview(curvature=curvature), empty_model())

  assert output.map_prediction.required_lateral_accel > 2.0
  assert V_EGO - output.map_prediction.safe_speed < 5.0 / 3.6
  assert not output.map_prediction.unsafe
  assert output.state == WarningState.idle
  assert not output.emit_event


def test_map_and_camera_rejection_reasons_remain_independently_diagnosable():
  invalid_camera = model_with_points({
    first_index_at_or_after(2.0): (math.nan, -1.0, 30.0),
  })
  output = PredictiveBendWarning(enabled=True).update(
    True,
    V_EGO,
    map_preview(valid=False, rejection_reason="ambiguousPath"),
    invalid_camera,
  )

  assert not output.map_valid
  assert output.map_prediction.rejection_reason == RejectionReason.ambiguousPath
  assert not output.vision_valid
  assert output.vision_prediction.rejection_reason == RejectionReason.invalidCurvature
  assert output.state == WarningState.idle


def test_valid_safe_source_clears_an_existing_warning():
  controller = PredictiveBendWarning(enabled=True)
  advance_to_warning(controller)

  output = controller.update(
    True, V_EGO, map_preview(curvature=SAFE_CURVATURE), empty_model(),
  )

  assert output.map_valid
  assert not output.map_prediction.unsafe
  assert output.state == WarningState.clearing
  assert output.emit_event
