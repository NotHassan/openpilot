"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import log, custom
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.params import Params
from openpilot.sunnypilot.mads.helpers import MadsSteeringModeOnBrake, read_steering_mode_param, MADS_NO_ACC_MAIN_BUTTON
from openpilot.sunnypilot.mads.state import StateMachine, GEARS_ALLOW_PAUSED_SILENT

State = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
ButtonType = structs.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
GearShifter = structs.CarState.GearShifter
SafetyModel = structs.CarParams.SafetyModel

SET_SPEED_BUTTONS = (ButtonType.accelCruise, ButtonType.resumeCruise, ButtonType.decelCruise, ButtonType.setCruise)
IGNORED_SAFETY_MODES = (SafetyModel.silent, SafetyModel.noOutput)

# Hands-on pause (param MadsPauseLateralOnHandsOn): pause lateral while the driver has hands on
# the wheel, resume shortly after release. Detection is capacitive touch
# (carState.steeringSlightlyPressed) OR sustained steering torque -- palm-on-wheel steering evades
# capacitive zones entirely, so torque is the fallback. Torque needs a longer sustain than touch
# because brief rack-recoil transients can exceed 1 Nm hands-off.
HANDS_ON_PAUSE_FRAMES = 30      # ~0.3s at 100Hz of touch before pausing
PALM_TORQUE = 100               # 1.0 Nm -- to TRIGGER a pause (assist active -> recoil transients exist)
PALM_TORQUE_HOLD = 40           # 0.4 Nm -- to HOLD a pause (no assist while paused -> torque is pure driver,
                                #           so even light palm guidance mid-turn keeps lateral paused)
PALM_TORQUE_PAUSE_FRAMES = 50   # ~0.5s at 100Hz of sustained torque before pausing
HANDS_OFF_RESUME_FRAMES = 30    # ~0.3s at 100Hz with neither before lateral resumes
# During a lane change the driver holds/nudges to guide the car over, so touch and same-direction
# torque must NOT pause (that cut steering out mid-maneuver). Distinguish intent by DIRECTION
# (steeringTorque > 0 = left, per desire_helper): steering back toward the original lane is
# "changing your mind" and aborts even gently; steering with the change guides; a firm grab either
# way is a full takeover. Pausing lateral drops lane_change_state to off, cleanly aborting.
LANE_CHANGE_ABORT_TORQUE = 80      # 0.8 Nm opposing the change -> gentle "change my mind" -> abort
LANE_CHANGE_TAKEOVER_TORQUE = 250  # 2.5 Nm either direction -> firm deliberate takeover
LANE_CHANGE_PAUSE_FRAMES = 15      # ~0.15s sustain to confirm a lane-change abort/takeover


class ModularAssistiveDrivingSystem:
  def __init__(self, selfdrive):
    self.CP = selfdrive.CP
    self.CP_SP = selfdrive.CP_SP
    self.params = selfdrive.params

    self.enabled = False
    self.active = False
    self.available = False
    self.lateral_mismatch_counter = 0
    self.hands_on_frames = 0      # TIGUAN: hands-on pause
    self.palm_frames = 0          # TIGUAN: hands-on pause
    self.hands_off_frames = 0     # TIGUAN: hands-on pause
    self.allow_always = False
    self.no_main_cruise = False
    self.selfdrive = selfdrive
    self.selfdrive.enabled_prev = False
    self.state_machine = StateMachine(self)
    self.events = self.selfdrive.events
    self.events_sp = self.selfdrive.events_sp
    self.disengage_on_accelerator = Params().get_bool("DisengageOnAccelerator")
    if self.CP.brand == "hyundai":
      if self.CP.flags & (HyundaiFlags.HAS_LDA_BUTTON | HyundaiFlags.CANFD):
        self.allow_always = True
    if self.CP.brand == "tesla":
      self.allow_always = True

    if self.CP.brand in MADS_NO_ACC_MAIN_BUTTON:
      self.no_main_cruise = True

    # read params on init
    self.enabled_toggle = self.params.get_bool("Mads")
    self.pause_on_hands_on = self.params.get_bool("MadsPauseLateralOnHandsOn")
    self.main_enabled_toggle = self.params.get_bool("MadsMainCruiseAllowed")
    self.steering_mode_on_brake = read_steering_mode_param(self.CP, self.CP_SP, self.params)
    self.unified_engagement_mode = self.params.get_bool("MadsUnifiedEngagementMode")

  def read_params(self):
    self.main_enabled_toggle = self.params.get_bool("MadsMainCruiseAllowed")
    self.pause_on_hands_on = self.params.get_bool("MadsPauseLateralOnHandsOn")
    self.unified_engagement_mode = self.params.get_bool("MadsUnifiedEngagementMode")

  def pedal_pressed_non_gas_pressed(self, CS: structs.CarState) -> bool:
    # ignore `pedalPressed` events caused by gas presses
    if self.events.has(EventName.pedalPressed) and not (CS.gasPressed and not self.selfdrive.CS_prev.gasPressed and self.disengage_on_accelerator):
      return True

    return False

  def should_silent_lkas_enable(self, CS: structs.CarState) -> bool:
    # TIGUAN: while the driver contacts the wheel (touch, or light torque -- see PALM_TORQUE_HOLD),
    # or released less than ~0.5s, stay paused
    if self.pause_on_hands_on and (CS.steeringSlightlyPressed or abs(CS.steeringTorque) > PALM_TORQUE_HOLD
                                   or self.hands_off_frames < HANDS_OFF_RESUME_FRAMES):
      return False

    if self.steering_mode_on_brake == MadsSteeringModeOnBrake.PAUSE and self.pedal_pressed_non_gas_pressed(CS):
      return False

    if self.events_sp.contains_in_list(GEARS_ALLOW_PAUSED_SILENT):
      return False

    return True

  def block_unified_engagement_mode(self) -> bool:
    # UEM disabled
    if not self.unified_engagement_mode:
      return True

    if self.enabled:
      return True

    if self.selfdrive.enabled and self.selfdrive.enabled_prev:
      return True

    return False

  def get_wrong_car_mode(self, alert_only: bool) -> None:
    if alert_only:
      if self.events.has(EventName.wrongCarMode):
        self.replace_event(EventName.wrongCarMode, EventNameSP.wrongCarModeAlertOnly)
    else:
      self.events.remove(EventName.wrongCarMode)

  def transition_paused_state(self):
    if self.state_machine.state != State.paused:
      self.events_sp.add(EventNameSP.silentLkasDisable)

  def replace_event(self, old_event: int, new_event: int):
    self.events.remove(old_event)
    self.events_sp.add(new_event)

  def data_sample(self):
    # When the safety and selfdrived do not agree on controls_allowed_lateral
    # we want to disengage sunnypilot. However the status from the panda goes through
    # another socket other than the CAN messages and one can arrive earlier than the other.
    # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
    if not self.active or self.selfdrive.enabled:
      self.lateral_mismatch_counter = 0
    elif any(not ps.controlsAllowedLateral for ps in self.selfdrive.sm['pandaStates']
             if ps.safetyModel not in IGNORED_SAFETY_MODES):
      self.lateral_mismatch_counter += 1

  def update_events(self, CS: structs.CarState):
    if not self.selfdrive.enabled and self.enabled:
      if CS.standstill:
        if self.events.has(EventName.doorOpen):
          self.replace_event(EventName.doorOpen, EventNameSP.silentDoorOpen)
          self.transition_paused_state()
        if self.events.has(EventName.seatbeltNotLatched):
          self.replace_event(EventName.seatbeltNotLatched, EventNameSP.silentSeatbeltNotLatched)
          self.transition_paused_state()
      if self.events.has(EventName.wrongGear) and (CS.vEgo < 2.5 or CS.gearShifter == GearShifter.reverse):
        self.replace_event(EventName.wrongGear, EventNameSP.silentWrongGear)
        self.transition_paused_state()
      if self.events.has(EventName.reverseGear):
        self.replace_event(EventName.reverseGear, EventNameSP.silentReverseGear)
        self.transition_paused_state()
      if self.events.has(EventName.brakeHold):
        self.replace_event(EventName.brakeHold, EventNameSP.silentBrakeHold)
        self.transition_paused_state()
      if self.events.has(EventName.parkBrake):
        self.replace_event(EventName.parkBrake, EventNameSP.silentParkBrake)
        self.transition_paused_state()

      if self.steering_mode_on_brake == MadsSteeringModeOnBrake.PAUSE:
        if self.pedal_pressed_non_gas_pressed(CS):
          self.transition_paused_state()

      self.events.remove(EventName.preEnableStandstill)
      self.events.remove(EventName.belowEngageSpeed)
      self.events.remove(EventName.speedTooLow)
      self.events.remove(EventName.cruiseDisabled)
      self.events.remove(EventName.manualRestart)
      self.events.remove(EventName.espActive)

    selfdrive_enable_events = self.events.has(EventName.pcmEnable) or self.events.has(EventName.buttonEnable)
    set_speed_btns_enable = any(be.type in SET_SPEED_BUTTONS for be in CS.buttonEvents)

    # wrongCarMode alert only or actively block control
    self.get_wrong_car_mode(selfdrive_enable_events or set_speed_btns_enable)

    if selfdrive_enable_events:
      if self.pedal_pressed_non_gas_pressed(CS):
        self.events_sp.add(EventNameSP.pedalPressedAlertOnly)

      if self.block_unified_engagement_mode():
        self.events.remove(EventName.pcmEnable)
        self.events.remove(EventName.buttonEnable)
    else:
      if self.main_enabled_toggle:
        if CS.cruiseState.available and not self.selfdrive.CS_prev.cruiseState.available:
          self.events_sp.add(EventNameSP.lkasEnable)

    for be in CS.buttonEvents:
      if be.type == ButtonType.cancel:
        if not self.selfdrive.enabled and self.selfdrive.enabled_prev:
          self.events_sp.add(EventNameSP.manualLongitudinalRequired)
      if be.type == ButtonType.lkas and be.pressed and (CS.cruiseState.available or self.allow_always):
        if self.enabled:
          if self.selfdrive.enabled:
            self.events_sp.add(EventNameSP.manualSteeringRequired)
          else:
            self.events_sp.add(EventNameSP.lkasDisable)
        else:
          self.events_sp.add(EventNameSP.lkasEnable)

    if not CS.cruiseState.available and not self.no_main_cruise:
      self.events.remove(EventName.buttonEnable)
      if self.selfdrive.CS_prev.cruiseState.available:
        self.events_sp.add(EventNameSP.lkasDisable)

    if self.steering_mode_on_brake == MadsSteeringModeOnBrake.DISENGAGE:
      if self.pedal_pressed_non_gas_pressed(CS):
        if self.enabled:
          self.events_sp.add(EventNameSP.lkasDisable)
        else:
          # block lkasEnable if being sent, then send pedalPressedAlertOnly event
          if self.events_sp.contains(EventNameSP.lkasEnable):
            self.events_sp.remove(EventNameSP.lkasEnable)
            self.events_sp.add(EventNameSP.pedalPressedAlertOnly)

    # TIGUAN: sustained touch OR sustained torque (palm steering) pauses lateral;
    # counters also feed should_silent_lkas_enable
    if self.pause_on_hands_on:
      lane_change_active = False
      lc_dir = LaneChangeDirection.none
      if self.selfdrive.sm.seen['modelV2']:
        meta = self.selfdrive.sm['modelV2'].meta
        lane_change_active = meta.laneChangeState != LaneChangeState.off
        lc_dir = meta.laneChangeDirection
      tq = CS.steeringTorque
      if lane_change_active:
        # steering back toward the original lane (opposing the change) = change of mind -> abort;
        # a firm grab either direction = takeover. Same-direction/light input just guides.
        opposing = ((lc_dir == LaneChangeDirection.left and tq < -LANE_CHANGE_ABORT_TORQUE) or
                    (lc_dir == LaneChangeDirection.right and tq > LANE_CHANGE_ABORT_TORQUE))
        touch = False  # capacitive touch is direction-ambiguous during a lane change; ignore it
        palm = opposing or abs(tq) > LANE_CHANGE_TAKEOVER_TORQUE
        pause_frames_needed = LANE_CHANGE_PAUSE_FRAMES
      else:
        touch = CS.steeringSlightlyPressed
        palm = abs(tq) > PALM_TORQUE
        pause_frames_needed = PALM_TORQUE_PAUSE_FRAMES
      holding = abs(tq) > PALM_TORQUE_HOLD  # light contact counts toward *staying* paused
      self.hands_on_frames = self.hands_on_frames + 1 if touch else 0
      self.palm_frames = self.palm_frames + 1 if palm else 0
      self.hands_off_frames = 0 if (touch or holding) else self.hands_off_frames + 1
      if self.enabled and (self.hands_on_frames >= HANDS_ON_PAUSE_FRAMES or self.palm_frames >= pause_frames_needed):
        self.transition_paused_state()

    if self.should_silent_lkas_enable(CS):
      if self.state_machine.state == State.paused:
        self.events_sp.add(EventNameSP.silentLkasEnable)

    if self.lateral_mismatch_counter >= 200:
      self.events_sp.add(EventNameSP.controlsMismatchLateral)

    self.events.remove(EventName.pcmDisable)
    self.events.remove(EventName.buttonCancel)
    self.events.remove(EventName.pedalPressed)
    self.events.remove(EventName.wrongCruiseMode)

  def update(self, CS: structs.CarState):
    if not self.enabled_toggle:
      return

    self.data_sample()

    self.update_events(CS)

    if not self.CP.passive and self.selfdrive.initialized:
      self.enabled, self.active = self.state_machine.update()

    # Copy of previous SelfdriveD states for MADS events handling
    self.selfdrive.enabled_prev = self.selfdrive.enabled
