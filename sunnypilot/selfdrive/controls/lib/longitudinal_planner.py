"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import messaging, custom
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.predictive_bend_warning import BendWarningOutput, PredictiveBendWarning
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventNameSP, EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


# PSD early-hint layer: nav road curvature sees bends hundreds of meters beyond the ~190 m camera
# horizon. A hint alone starts a BOUNDED trim (max ~10 km/h below the setpoint at hint onset) and,
# once acting, rides through the hinted bend and restores through the normal gradual walk -- never
# a snap release on disagreement (a sudden speed-up toward what the driver judges as a bend is
# dangerous). Vision remains the authority: when it confirms, its deeper target takes over via min().
HINT_MIN_V_EGO = 27.       # m/s (~100 km/h): the regime where the camera horizon is insufficient
HINT_MAX_TRIM_MS = 2.78    # ~10 km/h: an unconfirmed hint can never trim more than this


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP, mpc):
    self.events_sp = EventsSP()
    self.resolver = SpeedLimitResolver()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl(CP)
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()
    self.params = Params()
    bend_warning_enabled = self.params.get_bool("PredictiveBendWarning")
    self.predictive_bend_warning = PredictiveBendWarning(bend_warning_enabled)
    self.bend_warning_output = BendWarningOutput(enabled=bend_warning_enabled)
    self.frame = 0

    self.output_v_target = 0.
    self.output_a_target = 0.
    self.hint_anchor_v = 0.
    self.hint_was_active = False

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> tuple[float, float]:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm)

    # PSD early-hint (curve-type predictions only; carstate filters on this platform)
    pred_curve = CS.cruiseState.speedLimitPredicative
    hint_active = pred_curve > 0. and v_ego > HINT_MIN_V_EGO
    if hint_active and not self.hint_was_active:
      self.hint_anchor_v = v_cruise_cluster  # cap references the setpoint at hint onset (no ratchet)
    self.hint_was_active = hint_active
    hint_v = max(pred_curve, self.hint_anchor_v - HINT_MAX_TRIM_MS) if hint_active else 255.

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    # press-truth: the driver's stalk presses arrive as stock button events (ICBM's own injected
    # presses never do -- CAN controllers don't receive their own transmissions), so "who changed
    # the setpoint" needs no step-size guessing
    BET = structs.CarState.ButtonEvent.Type
    pressed = {be.type for be in CS.buttonEvents if be.pressed}
    acc_on = bool(CS.cruiseState.enabled)
    # primary source: the 100Hz-latched flags (button EVENTS are single-frame and this 20Hz
    # consumer drops ~80% of them -- road bug: driver presses were fought); events kept as fallback
    cs_sp = sm['carStateSP']
    btn_adjust = bool(cs_sp.userCruisePressLatched) or \
                 bool(pressed & ({BET.accelCruise, BET.decelCruise} |
                                 ({BET.setCruise, BET.resumeCruise} if acc_on else set())))
    btn_set_engage = bool(cs_sp.userSetEngagePressLatched) or ((not acc_on) and BET.setCruise in pressed)
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp,
                    curve_v_target=min(self.scc.vision.output_v_target, hint_v),
                    curve_active=self.scc.vision.is_active or hint_active,
                    acc_enabled=acc_on, user_btn_adjust=btn_adjust, user_btn_set_engage=btn_set_engage)

    targets = {
      LongitudinalPlanSource.cruise: (v_cruise, a_ego),
      LongitudinalPlanSource.sccVision: (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      LongitudinalPlanSource.sccMap: (self.scc.map.output_v_target, self.scc.map.output_a_target),
      LongitudinalPlanSource.speedLimitAssist: (self.sla.output_v_target, self.sla.output_a_target),
    }

    self.source = min(targets, key=lambda k: targets[k][0])
    self.output_v_target, self.output_a_target = targets[self.source]
    return self.output_v_target, self.output_a_target

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.predictive_bend_warning.enabled = self.params.get_bool("PredictiveBendWarning")

    CS = sm["carState"]
    self.bend_warning_output = self.predictive_bend_warning.update(
      sm["carControl"].latActive,
      CS.vEgo,
      CS.cruiseState.bendPreview,
      sm["modelV2"],
    )
    if self.bend_warning_output.emit_event:
      self.events_sp.add(EventNameSP.predictiveBendWarning)

    self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)
    self.frame += 1

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control
    dec = longitudinalPlanSP.dec
    dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
    dec.enabled = self.dec.enabled()
    dec.active = self.dec.active()

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = self.scc.vision.state
    sccVision.vTarget = float(self.scc.vision.output_v_target)
    sccVision.aTarget = float(self.scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(self.scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(self.scc.vision.max_pred_lat_acc)
    sccVision.enabled = self.scc.vision.is_enabled
    sccVision.active = self.scc.vision.is_active
    # Map Control
    sccMap = smartCruiseControl.map
    sccMap.state = self.scc.map.state
    sccMap.vTarget = float(self.scc.map.output_v_target)
    sccMap.aTarget = float(self.scc.map.output_a_target)
    sccMap.enabled = self.scc.map.is_enabled
    sccMap.active = self.scc.map.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(self.resolver.speed_limit)
    resolver.speedLimitLast = float(self.resolver.speed_limit_last)
    resolver.speedLimitFinal = float(self.resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(self.resolver.speed_limit_final_last)
    resolver.speedLimitValid = self.resolver.speed_limit_valid
    resolver.speedLimitLastValid = self.resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(self.resolver.speed_limit_offset)
    resolver.distToSpeedLimit = float(self.resolver.distance)
    resolver.source = self.resolver.source
    assist = speedLimit.assist
    assist.state = self.sla.state
    assist.enabled = self.sla.is_enabled
    assist.active = self.sla.is_active
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    # Predictive Bend Warning diagnostics
    bendWarning = longitudinalPlanSP.bendWarning
    bendWarning.enabled = self.bend_warning_output.enabled
    bendWarning.lateralActive = self.bend_warning_output.lateral_active
    bendWarning.state = self.bend_warning_output.state.name
    bendWarning.source = self.bend_warning_output.source.name
    bendWarning.mapValid = self.bend_warning_output.map_valid
    bendWarning.visionValid = self.bend_warning_output.vision_valid
    bendWarning.curvature = float(self.bend_warning_output.curvature)
    bendWarning.distance = float(self.bend_warning_output.distance)
    bendWarning.timeToBend = float(self.bend_warning_output.time_to_bend)
    bendWarning.requiredLateralAccel = float(self.bend_warning_output.required_lateral_accel)
    bendWarning.safeSpeed = float(self.bend_warning_output.safe_speed)
    bendWarning.currentSpeed = float(self.bend_warning_output.current_speed)
    bendWarning.candidateTime = float(self.bend_warning_output.candidate_time)
    bendWarning.episode = self.bend_warning_output.episode
    bendWarning.rejectionReason = self.bend_warning_output.rejection_reason.name

    pm.send('longitudinalPlanSP', plan_sp_send)
