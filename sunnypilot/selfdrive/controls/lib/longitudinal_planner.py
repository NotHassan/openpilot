"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import messaging, custom
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


# Steering-saturation feedback: when the lateral authority guard clips the steering command for
# this long, the car is at the physical limit of the turn RIGHT NOW -- trim speed regardless of
# what the path prediction says. Clip transients from rate limiting are shorter than the trigger.
SAT_CLIP_CURV = 2.5e-4         # |desired| - |applied| curvature indicating a real clip (~0.5 deg wheel)
SAT_UTILIZATION = 0.96         # engage on APPROACH (~2.3 of 2.4 m/s2): routine firm bends run ~2.2 and must not trigger
SAT_LAT_ACCEL_LIM = 2.4        # matches the opendbc steering guard
SAT_MIN_V = 14.                # m/s; city corners (angle-guard territory) are not this system's job
SAT_TRIGGER_FRAMES = 10        # 0.5 s at 20 Hz sustained before engaging
SAT_RELEASE_FRAMES = 20        # 1.0 s clean before releasing
SAT_TRIM_MS = 3.4              # target = current speed minus ~12 km/h, refreshed while saturated


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

    self.output_v_target = 0.
    self.output_a_target = 0.
    self.sat_frames = 0
    self.sat_clean_frames = 0
    self.sat_active = False
    self.sat_anchor_v = 0.  # v_ego when saturation triggered: the trim target anchors HERE and never
                            # chases v_ego down (a lead car slowing us mid-bend must not drag the
                            # setpoint with it -- that is the removed brake-assist failure mode)

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

    # Steering-saturation feedback (safety net for prediction misses): the steering guard clipping
    # means the rack is out of authority NOW; feed a speed trim through the same curve-assist
    # machinery so there is exactly one setpoint authority (baseline/restore/user-press semantics
    # all apply unchanged).
    lat_active = sm['carControl'].latActive
    desired_curv = abs(sm['carControl'].actuators.curvature)
    applied_curv = abs(sm['carOutput'].actuatorsOutput.curvature)
    # two ways in: hard evidence (guard clipping = at the limit) or high utilization of the
    # authority envelope (approaching it -- trim while margin still exists)
    lat_limit = SAT_LAT_ACCEL_LIM / max(v_ego ** 2, 1.0)
    near_limit = applied_curv > SAT_UTILIZATION * lat_limit
    clipped = lat_active and v_ego > SAT_MIN_V and ((desired_curv - applied_curv) > SAT_CLIP_CURV or near_limit)
    if clipped:
      self.sat_frames += 1
      self.sat_clean_frames = 0
    else:
      self.sat_clean_frames += 1
      if self.sat_clean_frames >= SAT_RELEASE_FRAMES:
        self.sat_frames = 0
        self.sat_active = False
    if self.sat_frames >= SAT_TRIGGER_FRAMES and not self.sat_active:
      self.sat_active = True
      self.sat_anchor_v = v_ego
    elif self.sat_active and self.sat_frames >= SAT_TRIGGER_FRAMES + 50:
      # still clipping 2.5 s after the last step: deepen by one more step from current speed
      self.sat_anchor_v = min(self.sat_anchor_v, v_ego)
      self.sat_frames = SAT_TRIGGER_FRAMES
    sat_v_target = (self.sat_anchor_v - SAT_TRIM_MS) if self.sat_active else 255.  # 255 = V_CRUISE_UNSET sentinel

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    # press-truth: the driver's stalk presses arrive as stock button events (ICBM's own injected
    # presses never do -- CAN controllers don't receive their own transmissions), so "who changed
    # the setpoint" needs no step-size guessing
    BET = structs.CarState.ButtonEvent.Type
    pressed = {be.type for be in CS.buttonEvents if be.pressed}
    acc_on = bool(CS.cruiseState.enabled)
    btn_adjust = bool(pressed & ({BET.accelCruise, BET.decelCruise} |
                                 ({BET.setCruise, BET.resumeCruise} if acc_on else set())))
    btn_set_engage = (not acc_on) and BET.setCruise in pressed
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp,
                    curve_v_target=min(self.scc.vision.output_v_target, sat_v_target),
                    curve_active=self.scc.vision.is_active or self.sat_active,
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
    self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)

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

    pm.send('longitudinalPlanSP', plan_sp_send)
