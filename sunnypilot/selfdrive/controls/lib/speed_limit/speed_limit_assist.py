"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

from cereal import custom, car
from openpilot.common.params import Params
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import PCM_LONG_REQUIRED_MAX_SET_SPEED, CONFIRM_SPEED_THRESHOLD
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.helpers import compare_cluster_target, set_speed_limit_assist_availability

ButtonType = car.CarState.ButtonEvent.Type
EventNameSP = custom.OnroadEventSP.EventName
SpeedLimitAssistState = custom.LongitudinalPlanSP.SpeedLimit.AssistState
SpeedLimitSource = custom.LongitudinalPlanSP.SpeedLimit.Source

ACTIVE_STATES = (SpeedLimitAssistState.active, SpeedLimitAssistState.adapting)
ENABLED_STATES = (SpeedLimitAssistState.preActive, SpeedLimitAssistState.pending, *ACTIVE_STATES)

DISABLED_GUARD_PERIOD = 0.5  # secs.
# secs. Time to wait after activation before considering temp deactivation signal.
PRE_ACTIVE_GUARD_PERIOD = {
  True: 15,
  False: 5,
}
SPEED_LIMIT_CHANGED_HOLD_PERIOD = 1  # secs. Time to wait after speed limit change before switching to preActive.

LIMIT_MIN_ACC = -1.5  # m/s^2 Maximum deceleration allowed for limit controllers to provide.
LIMIT_MAX_ACC = 1.0   # m/s^2 Maximum acceleration allowed for limit controllers to provide while active.
LIMIT_MIN_SPEED = 8.33  # m/s, Minimum speed limit to provide as solution on limit controllers.
LIMIT_SPEED_OFFSET_TH = -1.  # m/s Maximum offset between speed limit and current speed for adapting state.

# Auto mode (param SpeedLimitNonPcmAutoMode; non-PCM-long / ICBM cars): fully automatic, zero
# confirmations, user-first.
#  - the user's manual setpoint is law for the CURRENT zone: overriding pauses the assist and
#    records the zone's limit; re-announcements of the same limit (incl. through nav dropouts)
#    never re-assert
#  - an actual zone-value change re-engages automatically at limit+offset, no confirmation
#  - on initial engage, the user's chosen setpoint is respected until the first zone change

V_CRUISE_UNSET = 255.

# Curve assist (auto mode + SCC-Vision -> ICBM): trim the setpoint just enough to take the bend
# ahead within steering authority, then restore. The bend is re-evaluated every frame, so the
# working bound is a ROLLING floor relative to current speed: a real deep bend staircases down as
# the car slows (floor follows v_ego), while a phantom reading can only ever pull a small step
# before the next frames correct it. The absolute reduction cap is just a catastrophe backstop.
# No rolling floor: with big-step (+/-10) ICBM presses the design is to open the speed-setpoint
# gap immediately (measured: ACC decel scales with the gap, ~1.0 m/s2 at 29). The command goes
# straight to the bend's required speed; protection is the absolute backstop below plus the
# live-target rule (recomputed every frame, never below what the bend needs, released the moment
# the bend is makeable). A phantom reading costs only the decel of the ~1-2 s until frames
# correct, then big-step restore recovers.
CURVE_MAX_REDUCTION = {True: 45, False: 28}  # kph / mph below baseline (absolute backstop)
CURVE_MIN_V_EGO = 15.  # m/s (~54 km/h): curve trim only at road speed

CRUISE_BUTTONS_PLUS = (ButtonType.accelCruise, ButtonType.resumeCruise)
CRUISE_BUTTONS_MINUS = (ButtonType.decelCruise, ButtonType.setCruise)
CRUISE_BUTTON_CONFIRM_HOLD = 0.5  # secs.


class SpeedLimitAssist:
  _speed_limit_final_last: float
  _distance: float
  v_ego: float
  a_ego: float
  v_offset: float

  def __init__(self, CP: car.CarParams, CP_SP: custom.CarParamsSP):
    self.params = Params()
    self.CP = CP
    self.CP_SP = CP_SP
    self.frame = -1
    self.long_engaged_timer = 0
    self.pre_active_timer = 0
    self.is_metric = self.params.get_bool("IsMetric")
    set_speed_limit_assist_availability(self.CP, self.CP_SP, self.params)
    self.enabled = self.params.get("SpeedLimitMode", return_default=True) == Mode.assist
    self.non_pcm_auto_mode = self.params.get_bool("SpeedLimitNonPcmAutoMode")
    self.curve_assist_enabled = self._read_bool_safe("CurveSpeedAssist")
    self.long_enabled = False
    self.long_enabled_prev = False
    self.is_enabled = False
    self.is_active = False
    self.output_v_target = V_CRUISE_UNSET
    self.output_a_target = 0.
    self.v_ego = 0.
    self.a_ego = 0.
    self.v_offset = 0.
    self.target_set_speed_conv = 0
    self.prev_target_set_speed_conv = 0
    self.v_cruise_cluster = 0.
    self.v_cruise_cluster_prev = 0.
    self.v_cruise_cluster_conv = 0
    self.prev_v_cruise_cluster_conv = 0
    self._has_speed_limit = False
    self._speed_limit = 0.
    self._speed_limit_final_last = 0.
    self.speed_limit_prev = 0.
    self.speed_limit_final_last_conv = 0
    self.prev_speed_limit_final_last_conv = 0
    self._distance = 0.
    self.state = SpeedLimitAssistState.disabled
    self._state_prev = SpeedLimitAssistState.disabled
    self.pcm_op_long = CP.openpilotLongitudinalControl and CP.pcmCruise
    self.override_limit_conv = -1   # TIGUAN auto mode: zone value (conv units) the user overrode in; -1 = none
    # curve assist state (auto mode only)
    self._curve_v_target = float(V_CRUISE_UNSET)  # m/s, from SCC-Vision
    self._curve_active = False
    self.curve_engaged = False
    self.curve_restoring = False
    self.curve_target_conv = -1
    self.curve_baseline_conv = -1  # what to walk back to after the bend
    self.curve_user_cancelled = False  # user spoke mid-bend: latched until this bend episode ends
    self.acc_enabled = False       # stock ACC engaged (cruiseState.enabled)
    self.acc_enabled_prev = False
    self.curve_frozen_frames = 0   # frames spent frozen through a takeover/disengage
    # Zone-change offset carry (auto mode): on a DESCENDING zone change, keep the driver's current
    # relative offset (100 in an 80 -> 80 in a 60); ascending changes use the configured offset.
    self.carried_target_conv = -1
    self._target_change_frames = 99  # frames since the working target last changed
    self.curve_restore_to_conv = -1  # restore endpoint (baseline capped at the current zone target)

    self._plus_hold = 0.
    self._minus_hold = 0.
    self._last_carstate_ts = 0.

    # TODO-SP: SLA's own output_a_target for planner
    # Solution functions mapped to respective states
    self.acceleration_solutions = {
      SpeedLimitAssistState.disabled: self.get_current_acceleration_as_target,
      SpeedLimitAssistState.inactive: self.get_current_acceleration_as_target,
      SpeedLimitAssistState.preActive: self.get_current_acceleration_as_target,
      SpeedLimitAssistState.pending: self.get_current_acceleration_as_target,
      SpeedLimitAssistState.adapting: self.get_adapting_state_target_acceleration,
      SpeedLimitAssistState.active: self.get_active_state_target_acceleration,
    }

  @property
  def speed_limit_changed(self) -> bool:
    return self._has_speed_limit and bool(self._speed_limit != self.speed_limit_prev)

  @property
  def v_cruise_cluster_changed(self) -> bool:
    return bool(self.v_cruise_cluster_conv != self.prev_v_cruise_cluster_conv)

  @property
  def target_set_speed_confirmed(self) -> bool:
    return bool(self.v_cruise_cluster_conv == self.target_set_speed_conv)

  @property
  def v_cruise_cluster_below_confirm_speed_threshold(self) -> bool:
    return bool(self.v_cruise_cluster_conv < CONFIRM_SPEED_THRESHOLD[self.is_metric])

  def update_active_event(self, events_sp: EventsSP) -> None:
    if self.v_cruise_cluster_below_confirm_speed_threshold:
      events_sp.add(EventNameSP.speedLimitChanged)
    else:
      events_sp.add(EventNameSP.speedLimitActive)

  def get_v_target_from_control(self) -> float:
    speed_conv = CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS
    # curve assist has priority: it also operates in user-override zones (baseline = user setpoint)
    if not self.pcm_op_long:
      if self.curve_engaged and self.curve_target_conv > 0 and self.acc_enabled:
        return self.curve_target_conv * speed_conv
      if self.curve_restoring and self.curve_baseline_conv > 0:
        restore_to = self.curve_restore_to_conv if self.curve_restore_to_conv > 0 else self.curve_baseline_conv
        return restore_to * speed_conv

    if self._has_speed_limit:
      if self.pcm_op_long and self.is_enabled:
        return self._speed_limit_final_last
      if not self.pcm_op_long and self.is_active:
        if self.non_pcm_auto_mode and self.carried_target_conv > 0:
          return self.carried_target_conv * speed_conv
        return self._speed_limit_final_last

    # Fallback
    return V_CRUISE_UNSET

  # TODO-SP: SLA's own output_a_target for planner
  def get_a_target_from_control(self) -> float:
    return self.a_ego

  def update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.is_metric = self.params.get_bool("IsMetric")
      set_speed_limit_assist_availability(self.CP, self.CP_SP, self.params)
      self.enabled = self.params.get("SpeedLimitMode", return_default=True) == Mode.assist
      self.non_pcm_auto_mode = self.params.get_bool("SpeedLimitNonPcmAutoMode")
      self.curve_assist_enabled = self._read_bool_safe("CurveSpeedAssist")

  def _read_bool_safe(self, key: str) -> bool:
    try:
      return self.params.get_bool(key)
    except Exception:
      return False

  def update_car_state(self, CS: car.CarState) -> None:
    now = time.monotonic()
    self._last_carstate_ts = now

    for b in CS.buttonEvents:
      if not b.pressed:
        if b.type in CRUISE_BUTTONS_PLUS:
          self._plus_hold = max(self._plus_hold, now + CRUISE_BUTTON_CONFIRM_HOLD)
        elif b.type in CRUISE_BUTTONS_MINUS:
          self._minus_hold = max(self._minus_hold, now + CRUISE_BUTTON_CONFIRM_HOLD)

  def _get_button_release(self, req_plus: bool, req_minus: bool) -> bool:
    now = time.monotonic()
    if req_plus and now <= self._plus_hold:
      self._plus_hold = 0.
      return True
    elif req_minus and now <= self._minus_hold:
      self._minus_hold = 0.
      return True

    # expired
    if now > self._plus_hold:
      self._plus_hold = 0.
    if now > self._minus_hold:
      self._minus_hold = 0.
    return False

  def update_calculations(self, v_cruise_cluster: float) -> None:
    speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    self.v_cruise_cluster = v_cruise_cluster

    # Update current velocity offset (error)
    self.v_offset = self._speed_limit_final_last - self.v_ego

    self.speed_limit_final_last_conv = round(self._speed_limit_final_last * speed_conv)
    self.v_cruise_cluster_conv = round(self.v_cruise_cluster * speed_conv)

    cst_low, cst_high = PCM_LONG_REQUIRED_MAX_SET_SPEED[self.is_metric]
    pcm_long_required_max = cst_low if self._has_speed_limit and self.speed_limit_final_last_conv < CONFIRM_SPEED_THRESHOLD[self.is_metric] else \
                            cst_high
    pcm_long_required_max_set_speed_conv = round(pcm_long_required_max * speed_conv)

    self.target_set_speed_conv = pcm_long_required_max_set_speed_conv if self.pcm_op_long else self.speed_limit_final_last_conv
    if not self.pcm_op_long and self.non_pcm_auto_mode and self.carried_target_conv > 0:
      self.target_set_speed_conv = self.carried_target_conv
    self._target_change_frames = 0 if self.target_set_speed_conv != self.prev_target_set_speed_conv else self._target_change_frames + 1

  @property
  def apply_confirm_speed_threshold(self) -> bool:
    # below CST: always require user confirmation
    if self.v_cruise_cluster_below_confirm_speed_threshold:
      return True

    # at/above CST:
    # - new speed limit >= CST: auto change
    # - new speed limit < CST: user confirmation required
    return bool(self.speed_limit_final_last_conv < CONFIRM_SPEED_THRESHOLD[self.is_metric])

  def get_current_acceleration_as_target(self) -> float:
    return self.a_ego

  def get_adapting_state_target_acceleration(self) -> float:
    if self._distance > 0:
      return (self._speed_limit_final_last ** 2 - self.v_ego ** 2) / (2. * self._distance)

    return self.v_offset / float(ModelConstants.T_IDXS[CONTROL_N])

  def get_active_state_target_acceleration(self) -> float:
    return self.v_offset / float(ModelConstants.T_IDXS[CONTROL_N])

  def _update_confirmed_state(self):
    if self._has_speed_limit:
      if self.v_offset < LIMIT_SPEED_OFFSET_TH:
        self.state = SpeedLimitAssistState.adapting
      else:
        self.state = SpeedLimitAssistState.active
    else:
      self.state = SpeedLimitAssistState.pending

  def _update_non_pcm_long_confirmed_state(self) -> bool:
    if self.target_set_speed_confirmed:
      return True

    if self.state != SpeedLimitAssistState.preActive:
      return False

    req_plus, req_minus = compare_cluster_target(self.v_cruise_cluster, self._speed_limit_final_last, self.is_metric)

    return self._get_button_release(req_plus, req_minus)

  def update_state_machine_pcm_op_long(self):
    self.long_engaged_timer = max(0, self.long_engaged_timer - 1)
    self.pre_active_timer = max(0, self.pre_active_timer - 1)

    # ACTIVE, ADAPTING, PENDING, PRE_ACTIVE, INACTIVE
    if self.state != SpeedLimitAssistState.disabled:
      if not self.long_enabled or not self.enabled:
        self.state = SpeedLimitAssistState.disabled

      else:
        # ACTIVE
        if self.state == SpeedLimitAssistState.active:
          if self.v_cruise_cluster_changed:
            self.state = SpeedLimitAssistState.inactive
          elif self.speed_limit_changed and self.apply_confirm_speed_threshold:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)
          elif self._has_speed_limit and self.v_offset < LIMIT_SPEED_OFFSET_TH:
            self.state = SpeedLimitAssistState.adapting

        # ADAPTING
        elif self.state == SpeedLimitAssistState.adapting:
          if self.v_cruise_cluster_changed:
            self.state = SpeedLimitAssistState.inactive
          elif self.speed_limit_changed and self.apply_confirm_speed_threshold:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)
          elif self.v_offset >= LIMIT_SPEED_OFFSET_TH:
            self.state = SpeedLimitAssistState.active

        # PENDING
        elif self.state == SpeedLimitAssistState.pending:
          if self.target_set_speed_confirmed:
            self._update_confirmed_state()
          elif self.speed_limit_changed:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)

        # PRE_ACTIVE
        elif self.state == SpeedLimitAssistState.preActive:
          if self.target_set_speed_confirmed:
            self._update_confirmed_state()
          elif self.pre_active_timer <= 0:
            # Timeout - session ended
            self.state = SpeedLimitAssistState.inactive

        # INACTIVE
        elif self.state == SpeedLimitAssistState.inactive:
          pass

    # DISABLED
    elif self.state == SpeedLimitAssistState.disabled:
      if self.long_enabled and self.enabled:
        # start or reset preActive timer if initially enabled or manual set speed change detected
        if not self.long_enabled_prev or self.v_cruise_cluster_changed:
          self.long_engaged_timer = int(DISABLED_GUARD_PERIOD / DT_MDL)

        elif self.long_engaged_timer <= 0:
          if self.target_set_speed_confirmed:
            self._update_confirmed_state()
          elif self._has_speed_limit:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)
          else:
            self.state = SpeedLimitAssistState.pending

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES

    return enabled, active

  def update_state_machine_non_pcm_long(self):
    self.long_engaged_timer = max(0, self.long_engaged_timer - 1)
    self.pre_active_timer = max(0, self.pre_active_timer - 1)

    # ACTIVE, ADAPTING, PENDING, PRE_ACTIVE, INACTIVE
    if self.state != SpeedLimitAssistState.disabled:
      if not self.long_enabled or not self.enabled:
        self.state = SpeedLimitAssistState.disabled

      else:
        # ACTIVE
        if self.state == SpeedLimitAssistState.active:
          if self.v_cruise_cluster_changed:
            self.state = SpeedLimitAssistState.inactive

          elif self.speed_limit_changed and self.apply_confirm_speed_threshold:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)

        # PRE_ACTIVE
        elif self.state == SpeedLimitAssistState.preActive:
          if self._update_non_pcm_long_confirmed_state():
            self.state = SpeedLimitAssistState.active
          elif self.pre_active_timer <= 0:
            # Timeout - session ended
            self.state = SpeedLimitAssistState.inactive

        # INACTIVE
        elif self.state == SpeedLimitAssistState.inactive:
          if self.speed_limit_changed:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)
          elif self._update_non_pcm_long_confirmed_state():
            self.state = SpeedLimitAssistState.active

    # DISABLED
    elif self.state == SpeedLimitAssistState.disabled:
      if self.long_enabled and self.enabled:
        # start or reset preActive timer if initially enabled or manual set speed change detected
        if not self.long_enabled_prev or self.v_cruise_cluster_changed:
          self.long_engaged_timer = int(DISABLED_GUARD_PERIOD / DT_MDL)

        elif self.long_engaged_timer <= 0:
          if self._update_non_pcm_long_confirmed_state():
            self.state = SpeedLimitAssistState.active
          elif self._has_speed_limit:
            self.state = SpeedLimitAssistState.preActive
            self.pre_active_timer = int(PRE_ACTIVE_GUARD_PERIOD[self.pcm_op_long] / DT_MDL)
          else:
            self.state = SpeedLimitAssistState.inactive

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES

    return enabled, active

  def _limit_conv(self) -> int:
    speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    return round(self._speed_limit * speed_conv) if self._has_speed_limit else -1

  def _update_curve_assist(self) -> None:
    # Trim the setpoint for the bend ahead (SCC-Vision), independent of zone-override state, then
    # restore. Overrides target_set_speed_conv so _expected_walk_change() classifies ICBM's walk
    # toward (and back from) the curve speed as expected -- never as a user override.
    if not (self.curve_assist_enabled and self.non_pcm_auto_mode and self.enabled):
      # feature/mode off: hard reset
      self.curve_engaged = False
      self.curve_restoring = False
      self.curve_baseline_conv = -1
      self.curve_frozen_frames = 0
      return

    speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    zone_in_charge = self.state == SpeedLimitAssistState.active and self._has_speed_limit

    # Driver takeover (brake) disengages stock ACC -- and on this car openpilot itself
    # (carControl.enabled follows it) -- and the cluster setpoint reads 0/garbage; treat none of
    # that as user input. Freeze the curve/restore state and continue on re-engage, so the
    # original speed is still restored after the bend (road forensics: braking mid-bend wiped the
    # restore memory and the setpoint stayed at the trimmed value). Bounded: a takeover longer
    # than 30 s drops the memory -- restoring a stale baseline minutes later could be wrong.
    # (Skip the first frame back too -- the prev-setpoint tracker spans the disengagement.)
    if not (self.long_enabled and self.acc_enabled and self.acc_enabled_prev):
      if self.curve_engaged and not self._curve_active:
        self.curve_engaged = False   # bend ended during the takeover: what's left to do is restore
        self.curve_restoring = True
      # the driver adjusting the setpoint memory in standby (or SET-engaging a new speed) is law:
      # a memory change AWAY from our target drops the curve state so the restore can't fight it.
      # Toward-changes are the standby walker itself. Bounds guard against invalid readings
      # (memory reads 0/unset around the disengage edges).
      if self.curve_engaged or self.curve_restoring:
        cur, prev = self.v_cruise_cluster_conv, self.prev_v_cruise_cluster_conv
        if 20 < cur < 300 and 20 < prev < 300 and cur != prev:
          ref = self.curve_restore_to_conv if (self.curve_restoring and self.curve_restore_to_conv > 0) else self.curve_baseline_conv
          if ref > 0 and abs(cur - ref) > abs(prev - ref):
            self.curve_engaged = False
            self.curve_restoring = False
            self.curve_baseline_conv = -1
            self.curve_user_cancelled = True
            self.curve_frozen_frames = 0
            return
      if self.curve_engaged or self.curve_restoring:
        self.curve_frozen_frames += 1
        if self.curve_frozen_frames > int(30. / DT_MDL):
          self.curve_engaged = False
          self.curve_restoring = False
          self.curve_baseline_conv = -1
          self.curve_frozen_frames = 0
      return
    self.curve_frozen_frames = 0

    # the user speaking mid-curve (or mid-restore) wins for this bend: drop out entirely.
    # Expectation must reference OUR commanded target (curve or restore value) -- not the zone
    # target update_calculations just reset -- or ICBM's own walk past the midpoint reads as user.
    if (self.curve_engaged or self.curve_restoring) and self.v_cruise_cluster_changed:
      ref_conv = self.curve_target_conv if self.curve_engaged else self.curve_baseline_conv
      step_size = abs(self.v_cruise_cluster_conv - self.prev_v_cruise_cluster_conv)
      toward = abs(self.v_cruise_cluster_conv - ref_conv) <= abs(self.prev_v_cruise_cluster_conv - ref_conv)
      # ICBM itself now big-steps +/-10 toward the target; only larger jumps or away-steps are the user
      if step_size > 10 or not toward:
        self.curve_engaged = False
        self.curve_restoring = False
        self.curve_baseline_conv = -1
        self.curve_user_cancelled = True  # theirs until this bend is over
        return

    curve_conv = round(self._curve_v_target * speed_conv) if self._curve_active and self._curve_v_target < V_CRUISE_UNSET else -1

    if curve_conv > 0 and self.v_ego >= CURVE_MIN_V_EGO:
      if self.curve_user_cancelled:
        return  # user owns this bend; re-arm only after the curve episode ends
      if not self.curve_engaged:
        # baseline to restore: the zone target when the assist is in charge, else the user's setpoint.
        # A chained bend engaging mid-restore keeps the ORIGINAL baseline -- re-snapshotting the
        # half-restored setpoint would silently forget the user's speed across S-curves.
        if self.curve_restoring and self.curve_baseline_conv > 0:
          baseline = self.curve_baseline_conv
        else:
          baseline = self.target_set_speed_conv if zone_in_charge else self.v_cruise_cluster_conv
        if curve_conv <= baseline - 2:
          self.curve_engaged = True
          self.curve_restoring = False
          self.curve_baseline_conv = baseline
      elif zone_in_charge:
        self.curve_baseline_conv = self.target_set_speed_conv  # track zone changes mid-bend
      if self.curve_engaged:
        abs_floor = self.curve_baseline_conv - CURVE_MAX_REDUCTION[self.is_metric]
        capped = max(curve_conv, abs_floor)
        self.curve_target_conv = min(capped, self.curve_baseline_conv)
        self.target_set_speed_conv = self.curve_target_conv
    else:
      self.curve_user_cancelled = False  # bend episode over: re-arm for the next one
      if self.curve_engaged:  # bend done: walk back up
        self.curve_engaged = False
        self.curve_restoring = True
      if self.curve_restoring:
        # if a zone change during the bend lowered the working target below the baseline, restore
        # only up to it -- walking up to the old baseline just to walk back down is wrong
        zone_target = self.carried_target_conv if self.carried_target_conv > 0 else \
                      (self.speed_limit_final_last_conv if self._has_speed_limit else -1)
        restore_to = min(self.curve_baseline_conv, zone_target) if zone_target > 0 else self.curve_baseline_conv
        self.curve_restore_to_conv = restore_to
        if zone_in_charge or self.curve_baseline_conv <= 0 or self.v_cruise_cluster_conv >= restore_to:
          # zone target takes over naturally, or we are back at the driver's speed
          self.curve_restoring = False
          self.curve_baseline_conv = -1
        else:
          self.target_set_speed_conv = restore_to

  def _expected_walk_change(self) -> bool:
    # ICBM walks the setpoint in +/-1 presses and, when the target is far, +/-10 big-step presses
    # (which the cluster may round to a 10s multiple, so any step up to 10 toward the target is
    # plausibly ours). A jump larger than one big press, or any step moving AWAY from the target,
    # is the user.
    if abs(self.v_cruise_cluster_conv - self.prev_v_cruise_cluster_conv) > 10:
      return False
    # both distances reference the CURRENT target: comparing against the previous frame's target
    # misclassifies the first press after a zone-change handoff. A short grace window after a
    # target change also forgives an in-flight press toward the old target.
    toward = abs(self.v_cruise_cluster_conv - self.target_set_speed_conv) < \
             abs(self.prev_v_cruise_cluster_conv - self.target_set_speed_conv)
    # the grace window only forgives an in-flight press toward the PREVIOUS target -- a press away
    # from both targets is the user (road bug: a user +10 mid-restore was dragged back down)
    toward_prev = abs(self.v_cruise_cluster_conv - self.prev_target_set_speed_conv) < \
                  abs(self.prev_v_cruise_cluster_conv - self.prev_target_set_speed_conv)
    return toward or (self._target_change_frames < 15 and toward_prev)

  def _zone_change_carry(self, old_limit_conv: int, new_limit_conv: int, basis_conv: int) -> None:
    # Descending zone change: keep the driver's current relative offset (their setpoint or the
    # running target, whichever was in charge). Ascending: back to the configured offset.
    if 0 < new_limit_conv < old_limit_conv and basis_conv > 0:
      carried = new_limit_conv + (basis_conv - old_limit_conv)
      self.carried_target_conv = max(new_limit_conv, min(carried, new_limit_conv + 45))
    else:
      self.carried_target_conv = -1

  def update_state_machine_non_pcm_auto(self):
    # Auto mode -- see SpeedLimitNonPcmAutoMode note above. States: active / inactive / disabled.
    self.long_engaged_timer = max(0, self.long_engaged_timer - 1)
    limit_conv = self._limit_conv()

    if self.state != SpeedLimitAssistState.disabled:
      if not self.long_enabled or not self.enabled:
        self.state = SpeedLimitAssistState.disabled
        self.carried_target_conv = -1

      elif self.state == SpeedLimitAssistState.active:
        if self.v_cruise_cluster_changed and self.acc_enabled and self.acc_enabled_prev and not self._expected_walk_change():
          # the user spoke: their setpoint is law for this zone
          self.state = SpeedLimitAssistState.inactive
          self.override_limit_conv = limit_conv
          self.carried_target_conv = -1
        else:
          # zone value change while active: carry the running offset downward, configured upward
          speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
          prev_limit_conv = round(self.speed_limit_prev * speed_conv) if self.speed_limit_prev > 0 else -1
          if limit_conv > 0 and prev_limit_conv > 0 and limit_conv != prev_limit_conv:
            basis = self.carried_target_conv if self.carried_target_conv > 0 else self.prev_speed_limit_final_last_conv
            self._zone_change_carry(prev_limit_conv, limit_conv, basis)

      else:  # inactive (and any other non-active holdover)
        if limit_conv > 0 and self.override_limit_conv <= 0:
          self.override_limit_conv = limit_conv  # engaged before a zone was known: baseline it
        elif limit_conv > 0 and limit_conv != self.override_limit_conv:
          # real zone change: structure takes over. Descending, keep the driver's current offset
          # (their overridden setpoint relative to the old zone); ascending, configured offset.
          basis = self.curve_baseline_conv if (self.curve_engaged or self.curve_restoring) else self.v_cruise_cluster_conv
          self._zone_change_carry(self.override_limit_conv, limit_conv, basis)
          self.state = SpeedLimitAssistState.active

    elif self.long_enabled and self.enabled:
      if not self.long_enabled_prev:
        self.long_engaged_timer = int(DISABLED_GUARD_PERIOD / DT_MDL)
      elif self.long_engaged_timer <= 0:
        # respect the setpoint the user engaged with, until the first zone change
        self.state = SpeedLimitAssistState.inactive
        self.override_limit_conv = limit_conv

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES
    return enabled, active

  def update_events(self, events_sp: EventsSP) -> None:
    # Auto mode is zero-interaction by design: zone changes adjust silently (the limit is already
    # on screen). The chimed "Auto adjusting to speed limit" alerts are for the confirm-style modes.
    if self.non_pcm_auto_mode:
      return

    if self.state == SpeedLimitAssistState.preActive:
      events_sp.add(EventNameSP.speedLimitPreActive)

    if self.state == SpeedLimitAssistState.pending and self._state_prev != SpeedLimitAssistState.pending:
      events_sp.add(EventNameSP.speedLimitPending)

    if self.is_active:
      if self._state_prev not in ACTIVE_STATES:
        self.update_active_event(events_sp)

      # only notify if we acquire a valid speed limit
      # do not check has_speed_limit here
      elif self._speed_limit != self.speed_limit_prev:
        if self.speed_limit_prev <= 0:
          self.update_active_event(events_sp)
        elif self.speed_limit_prev > 0 and self._speed_limit > 0:
          self.update_active_event(events_sp)

  def update(self, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float, v_cruise_cluster: float, speed_limit: float,
             speed_limit_final_last: float, has_speed_limit: bool, distance: float, events_sp: EventsSP,
             curve_v_target: float = float(V_CRUISE_UNSET), curve_active: bool = False,
             acc_enabled: bool = True) -> None:
    self._curve_v_target = curve_v_target
    self._curve_active = curve_active
    self.acc_enabled = acc_enabled
    self.long_enabled = long_enabled
    self.v_ego = v_ego
    self.a_ego = a_ego

    self._has_speed_limit = has_speed_limit
    self._speed_limit = speed_limit
    self._speed_limit_final_last = speed_limit_final_last
    self._distance = distance

    self.update_params()
    self.update_calculations(v_cruise_cluster)
    self._update_curve_assist()

    self._state_prev = self.state
    if self.pcm_op_long:
      self.is_enabled, self.is_active = self.update_state_machine_pcm_op_long()
    elif self.non_pcm_auto_mode:
      self.is_enabled, self.is_active = self.update_state_machine_non_pcm_auto()
    else:
      self.is_enabled, self.is_active = self.update_state_machine_non_pcm_long()

    self.update_events(events_sp)

    # Update change tracking variables
    self.speed_limit_prev = self._speed_limit
    self.v_cruise_cluster_prev = self.v_cruise_cluster
    self.long_enabled_prev = self.long_enabled
    self.prev_target_set_speed_conv = self.target_set_speed_conv
    self.prev_v_cruise_cluster_conv = self.v_cruise_cluster_conv
    self.prev_speed_limit_final_last_conv = self.speed_limit_final_last_conv
    self.acc_enabled_prev = self.acc_enabled

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
