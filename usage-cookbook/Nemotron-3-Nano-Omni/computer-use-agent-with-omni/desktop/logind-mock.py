#!/usr/bin/env python3
"""
Minimal mock of org.freedesktop.login1 (systemd-logind) for GNOME Shell.

GNOME Shell's loginManager.js creates a LoginManagerSystemd proxy that calls:
  - GetSession('auto') -> returns a session object path
  - Session.Type property -> returns 'x11'
  - Manager signals: PrepareForSleep, PrepareForShutdown

This mock provides just enough of the D-Bus interface to prevent the
JS exception that crashes gnome-shell in containers without systemd.
"""

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib

BUS_NAME = "org.freedesktop.login1"
MANAGER_PATH = "/org/freedesktop/login1"
SESSION_PATH = "/org/freedesktop/login1/session/auto"
SEAT_PATH = "/org/freedesktop/login1/seat/seat0"
USER_PATH = "/org/freedesktop/login1/user/_1000"

MANAGER_IFACE = "org.freedesktop.login1.Manager"
SESSION_IFACE = "org.freedesktop.login1.Session"
SEAT_IFACE = "org.freedesktop.login1.Seat"
USER_IFACE = "org.freedesktop.login1.User"
PROP_IFACE = "org.freedesktop.DBus.Properties"


class MockSession(dbus.service.Object):
    def __init__(self, bus):
        super().__init__(bus, SESSION_PATH)

    @dbus.service.method(SESSION_IFACE, in_signature="b", out_signature="")
    def TakeControl(self, force):
        pass

    @dbus.service.method(SESSION_IFACE, in_signature="", out_signature="")
    def ReleaseControl(self):
        pass

    @dbus.service.method(SESSION_IFACE, in_signature="uu", out_signature="hb")
    def TakeDevice(self, major, minor):
        import os
        path = f"/dev/char/{major}:{minor}"
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError:
            fd = os.open("/dev/null", os.O_RDWR)
        return dbus.types.UnixFd(fd), dbus.Boolean(False)

    @dbus.service.method(SESSION_IFACE, in_signature="uu", out_signature="")
    def ReleaseDevice(self, major, minor):
        pass

    @dbus.service.method(SESSION_IFACE, in_signature="", out_signature="")
    def Activate(self):
        pass

    @dbus.service.signal(SESSION_IFACE, signature="ub")
    def PauseDevice(self, major, minor):
        pass

    @dbus.service.signal(SESSION_IFACE, signature="u")
    def ResumeDevice(self, major):
        pass

    @dbus.service.method(PROP_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        props = {
            "Id": dbus.String("auto", variant_level=1),
            "Name": dbus.String("kasm-user", variant_level=1),
            "User": dbus.Struct([dbus.UInt32(1000), dbus.ObjectPath(USER_PATH)], variant_level=1),
            "Seat": dbus.Struct([dbus.String("seat0"), dbus.ObjectPath(SEAT_PATH)], variant_level=1),
            "Type": dbus.String("x11", variant_level=1),
            "Class": dbus.String("user", variant_level=1),
            "Active": dbus.Boolean(True, variant_level=1),
            "State": dbus.String("active", variant_level=1),
            "Display": dbus.String(":1", variant_level=1),
            "Remote": dbus.Boolean(False, variant_level=1),
        }
        return props.get(prop, dbus.String("", variant_level=1))

    @dbus.service.method(PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        return {
            "Id": dbus.String("auto"),
            "Name": dbus.String("kasm-user"),
            "Type": dbus.String("x11"),
            "Class": dbus.String("user"),
            "Active": dbus.Boolean(True),
            "State": dbus.String("active"),
            "Display": dbus.String(":1"),
            "Remote": dbus.Boolean(False),
        }


class MockSeat(dbus.service.Object):
    def __init__(self, bus):
        super().__init__(bus, SEAT_PATH)

    @dbus.service.method(PROP_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        props = {
            "Id": dbus.String("seat0", variant_level=1),
            "CanGraphical": dbus.Boolean(True, variant_level=1),
            "CanMultiSession": dbus.Boolean(False, variant_level=1),
        }
        return props.get(prop, dbus.String("", variant_level=1))


class MockUser(dbus.service.Object):
    def __init__(self, bus):
        super().__init__(bus, USER_PATH)

    @dbus.service.method(PROP_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        props = {
            "Name": dbus.String("kasm-user", variant_level=1),
            "UID": dbus.UInt32(1000, variant_level=1),
            "State": dbus.String("active", variant_level=1),
        }
        return props.get(prop, dbus.String("", variant_level=1))


class MockManager(dbus.service.Object):
    def __init__(self, bus):
        super().__init__(bus, MANAGER_PATH)

    @dbus.service.method(MANAGER_IFACE, in_signature="s", out_signature="o")
    def GetSession(self, session_id):
        return dbus.ObjectPath(SESSION_PATH)

    @dbus.service.method(MANAGER_IFACE, in_signature="s", out_signature="o")
    def GetSessionByPID(self, pid):
        return dbus.ObjectPath(SESSION_PATH)

    @dbus.service.method(MANAGER_IFACE, in_signature="u", out_signature="o")
    def GetUser(self, uid):
        return dbus.ObjectPath(USER_PATH)

    @dbus.service.method(MANAGER_IFACE, in_signature="s", out_signature="o")
    def GetSeat(self, seat_id):
        return dbus.ObjectPath(SEAT_PATH)

    @dbus.service.method(MANAGER_IFACE, in_signature="", out_signature="a(susso)")
    def ListSessions(self):
        return [(dbus.String("auto"), dbus.UInt32(1000), dbus.String("kasm-user"),
                 dbus.String("seat0"), dbus.ObjectPath(SESSION_PATH))]

    @dbus.service.method(MANAGER_IFACE, in_signature="ssss", out_signature="h")
    def Inhibit(self, what, who, why, mode):
        import os
        r, w = os.pipe()
        return dbus.types.UnixFd(r)

    @dbus.service.method(MANAGER_IFACE, in_signature="", out_signature="s")
    def CanSuspend(self):
        return "no"

    @dbus.service.method(MANAGER_IFACE, in_signature="", out_signature="s")
    def CanHibernate(self):
        return "no"

    @dbus.service.method(MANAGER_IFACE, in_signature="", out_signature="s")
    def CanPowerOff(self):
        return "no"

    @dbus.service.method(MANAGER_IFACE, in_signature="", out_signature="s")
    def CanReboot(self):
        return "no"

    @dbus.service.method(PROP_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        props = {
            "IdleHint": dbus.Boolean(False, variant_level=1),
            "IdleSinceHint": dbus.UInt64(0, variant_level=1),
            "IdleSinceHintMonotonic": dbus.UInt64(0, variant_level=1),
            "PreparingForShutdown": dbus.Boolean(False, variant_level=1),
            "PreparingForSleep": dbus.Boolean(False, variant_level=1),
        }
        return props.get(prop, dbus.Boolean(False, variant_level=1))

    @dbus.service.method(PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        return {
            "IdleHint": dbus.Boolean(False),
            "PreparingForShutdown": dbus.Boolean(False),
            "PreparingForSleep": dbus.Boolean(False),
        }

    @dbus.service.signal(MANAGER_IFACE, signature="b")
    def PrepareForSleep(self, active):
        pass

    @dbus.service.signal(MANAGER_IFACE, signature="b")
    def PrepareForShutdown(self, active):
        pass


def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    bus_name = dbus.service.BusName(BUS_NAME, bus)

    MockManager(bus)
    MockSession(bus)
    MockSeat(bus)
    MockUser(bus)

    print(f"logind-mock: Registered {BUS_NAME} on system bus")
    loop = GLib.MainLoop()
    loop.run()


if __name__ == "__main__":
    main()
