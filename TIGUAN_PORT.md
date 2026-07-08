# 2025–26 North American VW Tiguan (Mk3) port

Community port of sunnypilot (infiniteCable2 fork) for the 3rd-gen NA Tiguan.
**Lateral (steering) only** — longitudinal control is locked behind Automotive
Ethernet on this generation; your stock ACC handles speed.

## Install
On the comma device setup screen choose **Custom Software** and enter:

    installer.comma.ai/NotHassan/25tiguan

## Select the car (required)
This generation returns a **zero VIN**, so automatic fingerprinting cannot
match. After install, SSH to the device and run:

    cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python -c "
    from openpilot.common.params import Params
    Params().put('CarPlatformBundle', {'platform':'VOLKSWAGEN_TIGUAN_MK3','make':'Volkswagen','brand':'volkswagen','model':'Tiguan','year':['2025','2026'],'name':'Volkswagen Tiguan 2025-26'})"

then reboot the device. Verify after a drive starts: the vehicle shows as
Volkswagen Tiguan 2025-26.

## What's included
- Platform + fingerprint + the EA_01 fix (without it the port permanently
  canErrors — NA Tiguans don't broadcast EA_01)
- Steering authority guards: this rack hard-faults (latched until you power
  cycle the CAR) past ~100° steering angle or ~2.5 m/s² lateral accel; the
  port backs off before both. Expect the car to steer wide on very tight
  roundabouts/ramps — help it by hand, that's by design
- Steering feel fixes: without them the wheel "forgets to turn" ~1×/second
  on bends (the torsion bar reads bend forces as driver override)
- Calibrated fault watchdog: this car's lateral stack has a benign 1 Hz
  status heartbeat that sits exactly on the stock alarm threshold

## Known limitations
- Lateral only, forever (hardware). Speed = stock ACC
- Occasional "Steering Fault May Be Imminent" can still appear in unusual
  conditions — informational on this platform, steering keeps working
- Tight-corner authority is limited by the rack itself, not the port

All changes are gated to the VOLKSWAGEN_TIGUAN_MK3 fingerprint — other cars
on this branch behave exactly like upstream.
