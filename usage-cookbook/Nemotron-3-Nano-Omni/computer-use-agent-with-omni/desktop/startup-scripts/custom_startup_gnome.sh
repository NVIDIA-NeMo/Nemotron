#!/bin/bash
# ---------------------------------------------------------------------------
# custom_startup_gnome.sh — Launch GNOME session + desktop Flask API
# ---------------------------------------------------------------------------

echo "--- GNOME + desktop API custom startup ---"

export DISPLAY=${DISPLAY:-:1}
export XDG_SESSION_TYPE=x11
export XDG_SESSION_DESKTOP=gnome
export XDG_CURRENT_DESKTOP=${XDG_CURRENT_DESKTOP:-GNOME}
export GDK_BACKEND=x11

# D-Bus session bus
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
    eval "$(dbus-launch --sh-syntax)"
    export DBUS_SESSION_BUS_ADDRESS
fi

# XDG runtime dir
XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export XDG_RUNTIME_DIR

dconf update 2>/dev/null || true

# Set resolution to 1920x1080
xrandr --output VNC-0 --mode 1920x1080 2>/dev/null || true

# Start gvfsd for Nautilus
( /usr/libexec/gvfsd || /usr/lib/gvfsd || /usr/lib/gvfs/gvfsd || true ) &>/dev/null &

clear_desktop_icons() {
    local desktop_dir="${HOME}/Desktop"
    [ -d "$desktop_dir" ] || return 0

    chmod 755 "$desktop_dir" 2>/dev/null || true
    find "$desktop_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
    gsettings set org.gnome.desktop.background show-desktop-icons false 2>/dev/null || true
    gsettings set org.gnome.shell.extensions.ding show-home false 2>/dev/null || true
    gsettings set org.gnome.shell.extensions.ding show-trash false 2>/dev/null || true
    gsettings set org.gnome.shell.extensions.ding show-volumes false 2>/dev/null || true
    gsettings set org.gnome.shell.extensions.ding show-network-volumes false 2>/dev/null || true
    gsettings set org.gnome.shell enabled-extensions "['ubuntu-dock@ubuntu.com', 'ubuntu-appindicators@ubuntu.com']" 2>/dev/null || true
}

# Keep the desktop canvas empty; apps remain available from the dock and app grid.
clear_desktop_icons

# Dismiss GNOME Activities overview after shell starts
(
    sleep 15
    xdotool key Escape 2>/dev/null || true
    sleep 2
    xdotool key Escape 2>/dev/null || true
) &

# Launch desktop Flask API in background (waits for gnome-shell)
(
    sleep 15
    echo "Starting desktop Flask API on port ${API_PORT:-5000}"
    cd /home/kasm-user/server
    PYTHON3=$(command -v python3.9 2>/dev/null || command -v python3 2>/dev/null || echo python3); exec $PYTHON3 main.py
) &

# Launch GNOME session (foreground)
# Check if systemd is actually running (PID 1), not just the directory mock
echo "Starting GNOME session on $DISPLAY"
if pidof systemd > /dev/null 2>&1 || dbus-send --system --print-reply --dest=org.freedesktop.systemd1 /org/freedesktop/systemd1 org.freedesktop.DBus.Peer.Ping > /dev/null 2>&1; then
    exec gnome-session --disable-acceleration-check 2>&1
else
    # No real systemd — start gnome-shell directly (GNOME 42+ requires systemd for session)
    echo "No systemd detected, starting gnome-shell directly"
    # Try --x11 first (GNOME 43+), fall back to plain gnome-shell if flag unsupported
    if gnome-shell --help 2>&1 | grep -q -- '--x11'; then
        exec gnome-shell --x11 2>&1
    else
        exec gnome-shell 2>&1
    fi
fi
