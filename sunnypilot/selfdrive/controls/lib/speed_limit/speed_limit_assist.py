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
    self.acc_enabled = False       # stock ACC engaged (cruiseState.enabled)
    self.acc_enabled_prev = False
    # Zone-change offset carry (auto mode): on a DESCENDING zone change, keep the driver's current
    # relative offset (100 in an 80 -> 80 in a 60); ascending changes use the configured offset.
    self.carried_target_conv = -1
    self.carried_for_limit_conv = -1  # the zone limit the carry was computed FOR: any other limit ignores it
    self.user_btn_frames = 0         # frames since a genuine driver stalk press (press-truth)
    self.stable_limit_conv = -1      # zone debounce: the confirmed working zone limit
    self.stable_final_ms = 0.        # and its target (limit + offset), in m/s
    self._pending_limit_conv = -1
    self._pending_frames = 0
    self._stable_limit_prev = -1

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

    if self._has_speed_limit:
      if self.pcm_op_long and self.is_enabled:
        return self._speed_limit_final_last
      if not self.pcm_op_long and self.is_active:
        if self.non_pcm_auto_mode and self._carried_valid():
          return self.carried_target_conv * speed_conv
        if self.non_pcm_auto_mode and self.stable_limit_conv > 0:
          return self.stable_final_ms
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

    self._update_zone_debounce()
    self.speed_limit_final_last_conv = round(self.stable_final_ms * speed_conv) if self.stable_limit_conv > 0 else 0
    self.v_cruise_cluster_conv = round(self.v_cruise_cluster * speed_conv)

    cst_low, cst_high = PCM_LONG_REQUIRED_MAX_SET_SPEED[self.is_metric]
    pcm_long_required_max = cst_low if self._has_speed_limit and self.speed_limit_final_last_conv < CONFIRM_SPEED_THRESHOLD[self.is_metric] else \
                            cst_high
    pcm_long_required_max_set_speed_conv = round(pcm_long_required_max * speed_conv)

    self.target_set_speed_conv = pcm_long_required_max_set_speed_conv if self.pcm_op_long else self.speed_limit_final_last_conv
    if not self.pcm_op_long and self.non_pcm_auto_mode and self._carried_valid():
      self.target_set_speed_conv = self.carried_target_conv

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
    # the DEBOUNCED zone limit: raw resolver values oscillate between adjacent zones on winding
    # roads (road replay: 60<->80 flapping for minutes); a value must hold ~2 s to become the
    # working zone. Raw dropouts/flaps never reach the state machine or the carry hooks.
    return self.stable_limit_conv

  def _update_zone_debounce(self) -> None:
    speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    raw = round(self._speed_limit * speed_conv) if self._has_speed_limit else -1
    if raw > 0 and raw == self.stable_limit_conv:
      self.stable_final_ms = self._speed_limit_final_last  # track offset changes within the zone
      self._pending_limit_conv = -1
      self._pending_frames = 0
    elif raw > 0:
      if raw == self._pending_limit_conv:
        self._pending_frames += 1
      else:
        self._pending_limit_conv = raw
        self._pending_frames = 1
      if self._pending_frames >= int(2.0 / DT_MDL):
        self.stable_limit_conv = raw
        self.stable_final_ms = self._speed_limit_final_last
        self._pending_limit_conv = -1
        self._pending_frames = 0
    # raw <= 0 (dropout): hold the stable zone, reset pending
    else:
      self._pending_limit_conv = -1
      self._pending_frames = 0

  def _expected_walk_change(self) -> bool:
    # ICBM walks the setpoint in +/-1 presses and, when the target is far, +/-10 big-step presses
    # (which the cluster may round to a 10s multiple, so any step up to 10 toward the target is
    # plausibly ours). A jump larger than one big press, or any step moving AWAY from the target,
    # is the user.
    # press-truth ONLY: the driver's presses are directly observable as stock button events and
    # ICBM's injected presses never appear there. Distance/direction heuristics used to guess --
    # and road-replay proved they can misclassify injected presses. No button, not the user.
    return self.user_btn_frames <= 0

  def _zone_change_carry(self, old_limit_conv: int, new_limit_conv: int, basis_conv: int) -> None:
    # Descending zone change: keep the driver's current relative offset (their setpoint or the
    # running target, whichever was in charge). Ascending: back to the configured offset.
    if 0 < new_limit_conv < old_limit_conv and basis_conv > 0:
      carried = new_limit_conv + (basis_conv - old_limit_conv)
      self.carried_target_conv = max(new_limit_conv, min(carried, new_limit_conv + 45))
      self.carried_for_limit_conv = new_limit_conv
    else:
      self.carried_target_conv = -1
      self.carried_for_limit_conv = -1

  def _carried_valid(self) -> bool:
    # a carried target is only meaningful in the exact zone it was computed for -- resolver
    # dropouts between zones can skip the zone-change hooks, and a stale carry then silently
    # becomes the working target (road bug: ascends never walked up; restores capped at odd
    # values like 82/93)
    return self.carried_target_conv > 0 and self._limit_conv() == self.carried_for_limit_conv

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
          prev_limit_conv = self._stable_limit_prev
          if limit_conv > 0 and prev_limit_conv > 0 and limit_conv != prev_limit_conv:
            basis = self.carried_target_conv if self._carried_valid() else self.prev_speed_limit_final_last_conv
            self._zone_change_carry(prev_limit_conv, limit_conv, basis)

      else:  # inactive (and any other non-active holdover)
        if limit_conv > 0 and self.override_limit_conv <= 0:
          self.override_limit_conv = limit_conv  # engaged before a zone was known: baseline it
        elif limit_conv > 0 and limit_conv != self.override_limit_conv:
          # real zone change: structure takes over. Descending, keep the driver's current offset
          # (their overridden setpoint relative to the old zone); ascending, configured offset.
          self._zone_change_carry(self.override_limit_conv, limit_conv, self.v_cruise_cluster_conv)
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
             acc_enabled: bool = True, user_btn_adjust: bool = False, user_btn_set_engage: bool = False) -> None:
    self.acc_enabled = acc_enabled
    if user_btn_adjust or user_btn_set_engage:
      self.user_btn_frames = int(1.0 / DT_MDL)   # driver spoke: latch for the cluster's reaction time
    else:
      self.user_btn_frames = max(0, self.user_btn_frames - 1)
    self.long_enabled = long_enabled
    self.v_ego = v_ego
    self.a_ego = a_ego

    self._has_speed_limit = has_speed_limit
    self._speed_limit = speed_limit
    self._speed_limit_final_last = speed_limit_final_last
    self._distance = distance

    self.update_params()
    self.update_calculations(v_cruise_cluster)

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
    self._stable_limit_prev = self.stable_limit_conv

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
