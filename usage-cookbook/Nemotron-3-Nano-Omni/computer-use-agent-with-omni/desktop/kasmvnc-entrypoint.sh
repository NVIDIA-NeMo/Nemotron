#!/bin/bash
# ---------------------------------------------------------------------------
# kasmvnc-entrypoint.sh — Wrapper entrypoint that starts system services
# needed by GNOME (D-Bus, logind mock) as root, then drops to kasm-user for
# the normal Kasm startup chain.
# ---------------------------------------------------------------------------

echo "[entrypoint] Starting kasmvnc-entrypoint.sh" >&2

# Generate machine-id if missing
if [ ! -s /etc/machine-id ]; then
    dbus-uuidgen > /etc/machine-id 2>/dev/null || cat /proc/sys/kernel/random/uuid | tr -d '-' > /etc/machine-id
    cp /etc/machine-id /var/lib/dbus/machine-id 2>/dev/null || true
fi

# Fix PAM on EL8/EL9: remove pam_systemd which hangs without real systemd
if [ -f /etc/pam.d/system-auth ]; then
    sed -i '/pam_systemd/d' /etc/pam.d/system-auth 2>/dev/null || true
    sed -i '/pam_systemd/d' /etc/pam.d/password-auth 2>/dev/null || true
fi

# Remove D-Bus service activation files that try to start systemd services
# These hang in containers without real systemd (both system and session buses)
rm -f /usr/share/dbus-1/system-services/org.freedesktop.RealtimeKit1.service 2>/dev/null || true
rm -f /usr/share/dbus-1/system-services/org.freedesktop.systemd1.service 2>/dev/null || true
rm -f /usr/share/dbus-1/services/org.freedesktop.systemd1.service 2>/dev/null || true

# Start system D-Bus (GNOME needs it for various org.freedesktop.* services)
mkdir -p /run/dbus
rm -f /run/dbus/pid /run/dbus/system_bus_socket 2>/dev/null || true
dbus-daemon --system --nofork --nopidfile &
sleep 0.3

# Mark system as "systemd-like" for timedatectl etc.
mkdir -p /run/systemd/system

# Start logind mock (GNOME shell requires org.freedesktop.login1)
PYTHON3_BIN=$(command -v python3.9 2>/dev/null || command -v python3 2>/dev/null || echo python3)
$PYTHON3_BIN /usr/local/bin/logind-mock.py &
sleep 0.5

echo "[entrypoint] Dropping to kasm-user..." >&2

# Drop to kasm-user preserving environment (VNC_RESOLUTION etc.)
# Set HOME explicitly so kasm-user doesn't inherit root's HOME
export HOME=/home/kasm-user

# Start session D-Bus as kasm-user (not root) so DEs can use it properly
DBUS_SOCKET="/tmp/dbus-session-kasm"
rm -f "$DBUS_SOCKET" 2>/dev/null || true
if command -v gosu &>/dev/null; then
    gosu kasm-user dbus-daemon --session --address="unix:path=$DBUS_SOCKET" --nofork --nopidfile &
else
    su -m -s /bin/bash kasm-user -c "dbus-daemon --session --address='unix:path=$DBUS_SOCKET' --nofork --nopidfile" &
fi
sleep 0.3
export DBUS_SESSION_BUS_ADDRESS="unix:path=$DBUS_SOCKET"

# Wrap dbus-launch to use our pre-started session bus instead of spawning new ones
DBUS_LAUNCH_REAL=$(command -v dbus-launch 2>/dev/null || echo "")
if [ -n "$DBUS_LAUNCH_REAL" ] && [ -x "$DBUS_LAUNCH_REAL" ]; then
    mv "$DBUS_LAUNCH_REAL" "${DBUS_LAUNCH_REAL}.real" 2>/dev/null || true
    cat > "$DBUS_LAUNCH_REAL" << 'DBUSWRAP'
#!/bin/bash
if [ -n "$DBUS_SESSION_BUS_ADDRESS" ]; then
    echo "DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS"
    echo "DBUS_SESSION_BUS_PID=0"
    exit 0
fi
exec "$(dirname "$0")/dbus-launch.real" "$@"
DBUSWRAP
    chmod +x "$DBUS_LAUNCH_REAL"
fi

# Drop privileges: gosu first (clean, no PAM), fallback to su -m
if command -v gosu &>/dev/null; then
    echo "[entrypoint] Using gosu" >&2
    exec gosu kasm-user /bin/bash -c \
        'exec /dockerstartup/kasm_default_profile.sh /dockerstartup/vnc_startup.sh /dockerstartup/kasm_startup.sh "$@"' \
        -- "$@"
else
    echo "[entrypoint] Using su -m" >&2
    exec su -m -s /bin/bash kasm-user -c \
        'exec /dockerstartup/kasm_default_profile.sh /dockerstartup/vnc_startup.sh /dockerstartup/kasm_startup.sh "$@"' \
        -- "$@"
fi
