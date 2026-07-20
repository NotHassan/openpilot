import inspect
from types import SimpleNamespace

import pytest

from cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.selfdrived.events import EVENTS, ET as UpstreamET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.predictive_bend_warning import (
  BendWarningOutput,
  PredictiveBendWarning,
  RejectionReason,
  WarningSource,
  WarningState,
)
from openpilot.sunnypilot.selfdrive.selfdrived.events import (
  EVENTS_SP,
  AlertSize,
  AlertStatus,
  AudibleAlert,
  ET,
  EventNameSP,
  EventsSP,
  VisualAlert,
)


class FakeSubMaster(dict):
  def all_checks(self, service_list=None):
    return True


class FakePubMaster:
  def send(self, service, message):
    self.service = service
    self.message = message


class NoOp:
  def update(self, *args):
    pass


class FakeParams:
  def __init__(self, enabled):
    self.enabled = enabled
    self.reads = 0

  def get_bool(self, key):
    assert key == "PredictiveBendWarning"
    self.reads += 1
    return self.enabled


class RecordingBendWarning:
  def __init__(self, output):
    self.enabled = True
    self.output = output
    self.calls = []

  def update(self, lateral_active, v_ego, map_preview, model_data):
    self.calls.append((lateral_active, v_ego, map_preview, model_data))
    return self.output


def make_sm(*, lateral_active=True, v_ego=31.0):
  preview = SimpleNamespace(valid=True, curvature=0.004, distance=100.0, rejectionReason="none")
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=[0.0, 0.12]),
    velocity=SimpleNamespace(x=[v_ego, 30.0]),
    position=SimpleNamespace(x=[0.0, 90.0]),
  )
  return FakeSubMaster({
    "carControl": SimpleNamespace(latActive=lateral_active),
    "carState": SimpleNamespace(vEgo=v_ego, cruiseState=SimpleNamespace(bendPreview=preview)),
    "modelV2": model,
  })


def make_update_planner(output, *, param_enabled=True):
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.events_sp = EventsSP()
  planner.dec = NoOp()
  planner.e2e_alerts_helper = NoOp()
  planner.params = FakeParams(param_enabled)
  planner.frame = 1
  planner.predictive_bend_warning = RecordingBendWarning(output)
  planner.bend_warning_output = BendWarningOutput()
  return planner


def test_schema_has_all_diagnostics_and_python_matching_enum_ordinals():
  assert "bendWarning" in custom.LongitudinalPlanSP.schema.fields

  BendWarning = custom.LongitudinalPlanSP.BendWarning
  assert dict(BendWarning.State.schema.enumerants) == {state.name: state.value for state in WarningState}
  assert dict(BendWarning.Source.schema.enumerants) == {source.name: source.value for source in WarningSource}
  assert dict(BendWarning.RejectionReason.schema.enumerants) == {
    reason.name: reason.value for reason in RejectionReason
  }

  assert set(BendWarning.schema.fields) == {
    "enabled",
    "lateralActive",
    "state",
    "source",
    "mapValid",
    "visionValid",
    "curvature",
    "distance",
    "timeToBend",
    "requiredLateralAccel",
    "safeSpeed",
    "currentSpeed",
    "candidateTime",
    "episode",
    "rejectionReason",
  }
  assert EventNameSP.schema.enumerants["predictiveBendWarning"] == 24


def test_predictive_bend_alert_is_warning_only_and_keeps_steering_active():
  event = EventNameSP.predictiveBendWarning
  assert set(EVENTS_SP[event]) == {ET.WARNING}

  alert = EVENTS_SP[event][ET.WARNING]
  assert alert.alert_text_1 == "Sharp Bend Ahead"
  assert alert.alert_text_2 == "Be Ready to Steer"
  assert alert.alert_status == AlertStatus.userPrompt
  assert alert.alert_size == AlertSize.mid
  assert alert.visual_alert == VisualAlert.steerRequired
  assert alert.audible_alert == AudibleAlert.prompt

  event_msg = EventsSP()
  event_msg.add(event)
  [serialized] = event_msg.to_msg()
  assert serialized.warning
  assert not serialized.softDisable
  assert not serialized.immediateDisable
  assert not serialized.userDisable
  assert not serialized.noEntry


def test_upstream_steer_saturated_alert_is_unchanged():
  upstream_event = log.OnroadEvent.EventName.schema.enumerants["steerSaturated"]
  assert upstream_event == 66
  assert set(EVENTS[upstream_event]) == {UpstreamET.WARNING}

  alert = EVENTS[upstream_event][UpstreamET.WARNING]
  assert alert.alert_text_1 == "Take Control"
  assert alert.alert_text_2 == "Turn Exceeds Steering Limit"
  assert alert.alert_status == AlertStatus.userPrompt
  assert alert.alert_size == AlertSize.mid
  assert alert.visual_alert == VisualAlert.steerRequired
  assert alert.audible_alert == AudibleAlert.promptRepeat


@pytest.mark.parametrize(
  ("state", "emit_event", "expected_event"),
  [
    (WarningState.idle, False, False),
    (WarningState.candidate, False, False),
    (WarningState.warning, True, True),
    (WarningState.clearing, True, True),
  ],
)
def test_planner_uses_lateral_gate_actual_speed_aligned_model_and_output_event(state, emit_event, expected_event):
  output = BendWarningOutput(
    enabled=True,
    lateral_active=True,
    state=state,
    source=WarningSource.both,
    emit_event=emit_event,
  )
  planner = make_update_planner(output)
  sm = make_sm(lateral_active=True, v_ego=31.0)

  planner.update(sm)

  assert planner.predictive_bend_warning.calls == [
    (sm["carControl"].latActive, sm["carState"].vEgo, sm["carState"].cruiseState.bendPreview, sm["modelV2"]),
  ]
  assert len(sm["modelV2"].orientationRate.z) == len(sm["modelV2"].velocity.x) == len(sm["modelV2"].position.x)
  assert planner.bend_warning_output is output
  assert planner.events_sp.has(EventNameSP.predictiveBendWarning) is expected_event


def test_parameter_refresh_and_lateral_inactive_reset_immediately():
  planner = make_update_planner(BendWarningOutput())
  planner.predictive_bend_warning = PredictiveBendWarning(enabled=False)
  planner.params.enabled = True
  planner.frame = 0
  sm = make_sm()

  planner.update(sm)
  assert planner.params.reads == 1
  assert planner.predictive_bend_warning.enabled

  planner.params.enabled = False
  planner.update(sm)
  assert planner.params.reads == 1
  assert planner.predictive_bend_warning.enabled

  planner.frame = int(PARAMS_UPDATE_PERIOD / DT_MDL)
  planner.update(sm)
  assert planner.params.reads == 2
  assert not planner.predictive_bend_warning.enabled
  assert planner.bend_warning_output.state == WarningState.idle
  assert planner.bend_warning_output.rejection_reason == RejectionReason.disabled
  assert not planner.events_sp.has(EventNameSP.predictiveBendWarning)

  planner.params.enabled = True
  planner.frame = int(PARAMS_UPDATE_PERIOD / DT_MDL) * 2
  planner.update(sm)
  assert planner.predictive_bend_warning.enabled
  planner.update(make_sm(lateral_active=False))
  assert planner.bend_warning_output.state == WarningState.idle
  assert planner.bend_warning_output.rejection_reason == RejectionReason.lateralInactive
  assert not planner.events_sp.has(EventNameSP.predictiveBendWarning)


def test_update_targets_remains_independent_of_predictive_bend_warning():
  source = inspect.getsource(LongitudinalPlannerSP.update_targets)
  assert "predictive_bend_warning" not in source
  assert "bend_warning_output" not in source


def test_publish_copies_every_diagnostic_field():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.source = custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise
  planner.output_v_target = 25.0
  planner.output_a_target = -0.2
  planner.events_sp = EventsSP()
  planner.dec = SimpleNamespace(mode=lambda: "acc", enabled=lambda: False, active=lambda: False)
  planner.scc = SimpleNamespace(
    vision=SimpleNamespace(
      state=custom.LongitudinalPlanSP.SmartCruiseControl.VisionState.enabled,
      output_v_target=25.0,
      output_a_target=-0.2,
      current_lat_acc=0.2,
      max_pred_lat_acc=1.0,
      is_enabled=False,
      is_active=False,
    ),
    map=SimpleNamespace(
      state=custom.LongitudinalPlanSP.SmartCruiseControl.MapState.enabled,
      output_v_target=25.0,
      output_a_target=-0.2,
      is_enabled=False,
      is_active=False,
    ),
  )
  planner.resolver = SimpleNamespace(
    speed_limit=0.0,
    speed_limit_last=0.0,
    speed_limit_final=0.0,
    speed_limit_final_last=0.0,
    speed_limit_valid=False,
    speed_limit_last_valid=False,
    speed_limit_offset=0.0,
    distance=0.0,
    source=custom.LongitudinalPlanSP.SpeedLimit.Source.none,
  )
  planner.sla = SimpleNamespace(
    state=custom.LongitudinalPlanSP.SpeedLimit.AssistState.disabled,
    is_enabled=False,
    is_active=False,
    output_v_target=25.0,
    output_a_target=-0.2,
  )
  planner.e2e_alerts_helper = SimpleNamespace(green_light_alert=False, lead_depart_alert=False)
  planner.bend_warning_output = BendWarningOutput(
    enabled=True,
    lateral_active=True,
    state=WarningState.warning,
    source=WarningSource.both,
    map_valid=True,
    vision_valid=True,
    curvature=-0.004,
    distance=123.0,
    time_to_bend=4.1,
    required_lateral_accel=3.2,
    safe_speed=22.3,
    current_speed=30.0,
    candidate_time=0.5,
    episode=7,
    rejection_reason=RejectionReason.none,
    emit_event=True,
  )
  pm = FakePubMaster()

  planner.publish_longitudinal_plan_sp(FakeSubMaster(), pm)

  diagnostics = pm.message.longitudinalPlanSP.bendWarning
  assert diagnostics.enabled
  assert diagnostics.lateralActive
  assert diagnostics.state == "warning"
  assert diagnostics.source == "both"
  assert diagnostics.mapValid
  assert diagnostics.visionValid
  assert diagnostics.curvature == pytest.approx(-0.004)
  assert diagnostics.distance == pytest.approx(123.0)
  assert diagnostics.timeToBend == pytest.approx(4.1)
  assert diagnostics.requiredLateralAccel == pytest.approx(3.2)
  assert diagnostics.safeSpeed == pytest.approx(22.3)
  assert diagnostics.currentSpeed == pytest.approx(30.0)
  assert diagnostics.candidateTime == pytest.approx(0.5)
  assert diagnostics.episode == 7
  assert diagnostics.rejectionReason == "none"
