import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller import (
  IntelligentCruiseButtonManagement,
  SendButtonState,
  State,
)
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlanSource,
  LongitudinalPlannerSP,
)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import vision_controller
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist


OLD_SETTING = "Curve" + "SpeedAssist"
OLD_TARGET_ARG = "curve" + "_v_target"
OLD_ACTIVE_ARG = "curve" + "_active"
WARNING_LABEL = "Predictive Bend Warning"
WARNING_DESCRIPTION = "Warns when a bend ahead may exceed the Comma's steering capability at your current speed. The Comma keeps steering until you take over."


class _FakeController:
  def __init__(self, *, v_target=255.0, a_target=0.0, active=False):
    self.output_v_target = v_target
    self.output_a_target = a_target
    self.is_active = active

  def update(self, *args):
    pass


class _FakeScc:
  def __init__(self):
    self.vision = _FakeController()
    self.map = _FakeController()

  def update(self, *args):
    pass


class _FakeResolver:
  speed_limit_valid = False
  speed_limit_last_valid = False
  speed_limit = 0.0
  speed_limit_final_last = 0.0
  distance = 0.0

  def update(self, *args):
    pass


class _BendSensitiveFakeSla:
  """Expose any forbidden planner-to-SLA bend control as an observable target change."""

  def __init__(self):
    self.output_v_target = 255.0
    self.output_a_target = 0.0

  def update(self, *args, **kwargs):
    if kwargs.get(OLD_ACTIVE_ARG):
      self.output_v_target = kwargs[OLD_TARGET_ARG]
      self.output_a_target = -1.0


def _planner_target(predictive_speed):
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = _FakeScc()
  planner.resolver = _FakeResolver()
  planner.sla = _BendSensitiveFakeSla()
  planner.events_sp = object()
  planner.source = LongitudinalPlanSource.cruise
  planner.hint_anchor_v = 0.0
  planner.hint_was_active = False

  cruise_state = SimpleNamespace(
    enabled=True,
    speedLimitPredicative=predictive_speed,
    bendPreview=SimpleNamespace(valid=predictive_speed > 0),
  )
  sm = {
    "carState": SimpleNamespace(vCruiseCluster=110.0, cruiseState=cruise_state, buttonEvents=[]),
    "carStateSP": SimpleNamespace(userCruisePressLatched=False, userSetEngagePressLatched=False),
    "carControl": SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
  }
  return planner.update_targets(sm, v_ego=31.0, a_ego=0.2, v_cruise=110.0 / 3.6)


def test_bend_preview_cannot_change_planner_speed_or_acceleration_targets():
  safe_targets = _planner_target(0.0)
  unsafe_targets = _planner_target(20.0)

  assert unsafe_targets == safe_targets


def test_speed_limit_assist_has_no_bend_control_arguments():
  parameters = inspect.signature(SpeedLimitAssist.update).parameters

  assert OLD_TARGET_ARG not in parameters
  assert OLD_ACTIVE_ARG not in parameters


def test_bend_diagnostics_cannot_request_an_icbm_button():
  car_state = SimpleNamespace(cruiseState=SimpleNamespace(speedCluster=30.0))
  safe_plan = SimpleNamespace(vTarget=30.0)
  warning_plan = SimpleNamespace(
    vTarget=30.0,
    bendWarning=SimpleNamespace(state="warning", safeSpeed=15.0, requiredLateralAccel=4.0),
  )

  def requested_button(plan):
    controller = IntelligentCruiseButtonManagement.__new__(IntelligentCruiseButtonManagement)
    controller.is_metric = True
    controller.speed_limit_only = False
    controller.v_target_ms_last = 0.0
    controller.is_ready = True
    controller.is_ready_prev = False
    controller.state = State.inactive
    controller.pre_active_timer = 0
    controller.update_calculations(car_state, plan)
    controller.update_state_machine()
    controller.is_ready_prev = True
    controller.pre_active_timer = 0
    return controller.update_state_machine()

  assert requested_button(safe_plan) == SendButtonState.none
  assert requested_button(warning_plan) == SendButtonState.none


def test_icbm_does_not_walk_the_setpoint_while_cruise_is_disengaged():
  controller = IntelligentCruiseButtonManagement.__new__(IntelligentCruiseButtonManagement)
  controller.cruise_button_timers = {}
  car_state = SimpleNamespace(
    buttonEvents=[],
    cruiseState=SimpleNamespace(available=True, enabled=False),
  )
  car_control = SimpleNamespace(
    enabled=False,
    cruiseControl=SimpleNamespace(override=False, cancel=False, resume=False),
  )

  controller.update_readiness(car_state, car_control)

  assert not controller.is_ready


def test_warning_parameter_is_default_off_and_old_setting_is_not_registered():
  registry = Path("common/params_keys.h").read_text()
  params = Params()
  params.remove("PredictiveBendWarning")

  assert registry.count('{"PredictiveBendWarning", {PERSISTENT | BACKUP, BOOL, "0"}},') == 1
  assert OLD_SETTING not in registry
  assert params.get("PredictiveBendWarning", return_default=True) is False


def test_vision_enablement_depends_only_on_its_own_setting(monkeypatch):
  class FakeParams:
    def get_bool(self, key):
      return key == OLD_SETTING

  monkeypatch.setattr(vision_controller, "Params", FakeParams)

  assert not vision_controller.SmartCruiseControlVision().enabled


def test_smart_cruise_constructors_have_upstream_signatures():
  assert list(inspect.signature(SmartCruiseControl).parameters) == []
  assert list(inspect.signature(vision_controller.SmartCruiseControlVision).parameters) == []


def test_replacement_setting_copy_and_lateral_only_availability():
  standard_ui = Path("selfdrive/ui/layouts/settings/ictoggles.py").read_text()
  mici_ui = Path("selfdrive/ui/mici/layouts/settings/ictoggles.py").read_text()
  ui_state = Path("selfdrive/ui/sunnypilot/ui_state.py").read_text()
  settings = json.loads(Path("sunnypilot/sunnylink/settings_ui.json").read_text())

  def find_setting(value):
    if isinstance(value, dict):
      if value.get("key") == "PredictiveBendWarning":
        return value
      return next((found for child in value.values() if (found := find_setting(child)) is not None), None)
    if isinstance(value, list):
      return next((found for child in value if (found := find_setting(child)) is not None), None)
    return None

  sunnylink_setting = find_setting(settings)
  assert sunnylink_setting is not None
  assert sunnylink_setting["title"] == WARNING_LABEL
  assert sunnylink_setting["description"] == WARNING_DESCRIPTION
  assert WARNING_LABEL in standard_ui
  assert WARNING_DESCRIPTION in standard_ui
  assert f'BigParamControl("{WARNING_LABEL}", "PredictiveBendWarning")' in mici_ui
  assert 'remove("PredictiveBendWarning")' not in ui_state
