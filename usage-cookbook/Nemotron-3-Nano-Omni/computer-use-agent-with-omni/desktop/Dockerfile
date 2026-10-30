# ===========================================================================
# Computer Use Agent with Omni desktop - KasmVNC + GNOME + minimal desktop API
# Based on the ProRL-Agent-Server desktop recipe, with a reduced package and
# API surface for this demo.
#
# Base: kasmweb/ubuntu-jammy-desktop pinned by digest (KasmVNC + Xvnc built-in)
# DE: GNOME Shell
# Apps: Chrome, Firefox, LibreOffice, GIMP, VLC, VS Code, Thunderbird
# API: Minimal Flask server on port 5000
#
# Build: docker build -t computer-use-agent-desktop ./desktop
# Run:   docker run -d --shm-size=2g computer-use-agent-desktop
# ===========================================================================
FROM kasmweb/ubuntu-jammy-desktop@sha256:288de2b78d30d42dba80a9b3c806459762f8d404c3b851ad7483e030a9dad1d8

USER root

# ===========================================================================
# 1. Remove default XFCE (Kasm core ships with XFCE)
# ===========================================================================
RUN rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
    && apt-get remove -y --purge \
        xfce4 xfce4-session xfce4-panel xfce4-settings xfce4-terminal \
        xfce4-appfinder xfce4-notifyd xfce4-screensaver xfce4-helpers \
        xfce4-pulseaudio-plugin \
        xfdesktop4 xfdesktop4-data xfwm4 thunar \
        libxfce4ui-2-0 libxfce4ui-common libxfce4ui-utils \
        libxfce4panel-2.0-4 libxfce4util7 libxfce4util-bin libxfce4util-common \
        libgarcon-1-0 libgarcon-gtk3-1-0 libexo-2-0 \
        xfconf libxfconf-0-3 gir1.2-xfconf-0 gir1.2-libxfce4util-1.0 \
        elementary-xfce-icon-theme 2>/dev/null \
    || true; \
    apt-mark manual code firefox google-chrome-stable \
        libreoffice-writer libreoffice-calc libreoffice-impress libreoffice-gtk3 \
        vlc gimp thunderbird 2>/dev/null || true; \
    apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ===========================================================================
# 2. System dependencies
# ===========================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        apt-transport-https ca-certificates curl wget gnupg \
        software-properties-common lsb-release \
        locales tzdata \
        dbus dbus-x11 dbus-user-session \
        python3 python3-pip python3-tk \
        python3-gi python3-gi-cairo python3-dbus \
        gir1.2-gtk-3.0 \
        xdotool xclip xsel xterm wmctrl \
        x11-xserver-utils \
        scrot \
        librsvg2-common libgdk-pixbuf2.0-bin \
        fonts-dejavu fonts-liberation fonts-noto fonts-noto-cjk fontconfig \
        xfonts-base xfonts-100dpi xfonts-75dpi \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ===========================================================================
# 3. GNOME Shell desktop
# ===========================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnome-shell \
        gnome-session \
        gnome-terminal \
        gnome-settings-daemon \
        gnome-shell-extension-ubuntu-dock \
        gnome-shell-extension-appindicator \
        gnome-shell-extension-desktop-icons-ng \
        gnome-software yelp \
        nautilus gvfs gvfs-backends \
        mutter glib-networking \
        dconf-cli dconf-service \
        adwaita-icon-theme adwaita-icon-theme-full \
        yaru-theme-gnome-shell yaru-theme-gtk yaru-theme-icon yaru-theme-sound \
        gnome-themes-extra \
        gsettings-desktop-schemas \
        xdg-user-dirs xdg-utils \
    && sed -i 's/^Icon=.*/Icon=software-store/' /usr/share/applications/org.gnome.Software.desktop \
    && rm -rf /var/lib/apt/lists/*

# ===========================================================================
# 4. Applications
# ===========================================================================

# Google Chrome
RUN install -d -m 0755 /etc/apt/keyrings \
    && wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
       | gpg --dearmor --batch --yes -o /etc/apt/keyrings/google-linux-signing-keyring.gpg \
    && chmod 0644 /etc/apt/keyrings/google-linux-signing-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-linux-signing-keyring.gpg] https://dl.google.com/linux/chrome/deb/ stable main" \
       > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -f /etc/apt/sources.list.d/google-chrome.sources \
    && rm -rf /var/lib/apt/lists/*

# LibreOffice, VLC, GIMP, Thunderbird
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer libreoffice-calc libreoffice-impress libreoffice-gtk3 \
        vlc vlc-plugin-access-extra \
        gimp gimp-data \
        thunderbird \
    && rm -rf /var/lib/apt/lists/* \
    && echo "/usr/lib/libreoffice/program" > /etc/ld.so.conf.d/libreoffice.conf \
    && ldconfig

# VS Code
RUN install -d -m 0755 /etc/apt/keyrings \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
       | gpg --dearmor --batch --yes -o /etc/apt/keyrings/microsoft.gpg \
    && chmod 0644 /etc/apt/keyrings/microsoft.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" \
       > /etc/apt/sources.list.d/vscode.list \
    && apt-get update && apt-get install -y code \
    && rm -rf /var/lib/apt/lists/*

# ===========================================================================
# 5. Python packages for the desktop API
# ===========================================================================
COPY requirements-desktop-api.txt /tmp/requirements-desktop-api.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements-desktop-api.txt 2>/dev/null \
    || pip3 install --no-cache-dir -r /tmp/requirements-desktop-api.txt

# ===========================================================================
# 6. User setup (kasm-user already exists from base image)
# ===========================================================================
RUN echo "kasm-user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && usermod -aG sudo,audio,video kasm-user \
    && ln -sf /home/kasm-user /home/user
RUN su - kasm-user -c "LC_ALL=C xdg-user-dirs-update"

# ===========================================================================
# 7. Application wrappers
# ===========================================================================

# Chrome wrapper
RUN cat > /usr/local/bin/google-chrome << 'CHROMEWRAP'
#!/bin/bash
set -euo pipefail
CHROME_BIN="/usr/bin/google-chrome-stable"
CHROME_DIR="${HOME}/.config/google-chrome-debug"
has_user_data_dir=false
for arg in "$@"; do
    [[ "$arg" == --user-data-dir=* ]] && has_user_data_dir=true
done
mkdir -p "$CHROME_DIR"
rm -f "$CHROME_DIR/SingletonLock" "$CHROME_DIR/SingletonSocket" "$CHROME_DIR/SingletonCookie" || true
extra_args=(--no-sandbox --no-first-run --no-default-browser-check --start-maximized
    --enable-webgl --ignore-gpu-blocklist --use-gl=swiftshader --enable-unsafe-swiftshader)
$has_user_data_dir || extra_args+=(--user-data-dir="$CHROME_DIR")
exec "$CHROME_BIN" "${extra_args[@]}" "$@"
CHROMEWRAP
RUN chmod +x /usr/local/bin/google-chrome \
    && ln -sf /usr/local/bin/google-chrome /usr/local/bin/google-chrome-wrapper

RUN cat > /usr/share/applications/google-chrome.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=Google Chrome
Exec=/usr/local/bin/google-chrome %U
StartupNotify=true
Terminal=false
Icon=google-chrome
Type=Application
Categories=Network;WebBrowser;
EOF

# VS Code wrapper
RUN cat > /usr/local/bin/code << 'CODEWRAP'
#!/bin/bash
export DONT_PROMPT_WSL_INSTALL=1
exec /usr/bin/code --no-sandbox "$@"
CODEWRAP
RUN chmod +x /usr/local/bin/code \
    && ln -sf /usr/local/bin/code /usr/local/bin/code-wrapper \
    && sed -i 's|Exec=/usr/share/code/code|Exec=/usr/local/bin/code|g' /usr/share/applications/code.desktop 2>/dev/null || true \
    && sed -i 's|Exec=code|Exec=/usr/local/bin/code|g' /usr/share/applications/code.desktop 2>/dev/null || true

# ===========================================================================
# 8. Application configs
# ===========================================================================
RUN mkdir -p /home/kasm-user/.config/vlc \
    && printf '[qt]\nqt-privacy-ask=0\n' > /home/kasm-user/.config/vlc/vlcrc
RUN mkdir -p /home/kasm-user/.config/Code/User \
    && printf '{"window.zoomLevel":0,"editor.fontSize":14,"telemetry.telemetryLevel":"off","update.mode":"none"}\n' \
       > /home/kasm-user/.config/Code/User/settings.json
RUN mkdir -p /home/kasm-user/.config/libreoffice/4/user \
    && printf '<?xml version="1.0" encoding="UTF-8"?>\n<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>\n<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" oor:op="fuse"><value>false</value></prop></item>\n</oor:items>\n' \
       > /home/kasm-user/.config/libreoffice/4/user/registrymodifications.xcu
RUN mkdir -p /home/kasm-user/.config/google-chrome-debug \
    && rm -rf /home/kasm-user/.config/google-chrome \
    && ln -s /home/kasm-user/.config/google-chrome-debug /home/kasm-user/.config/google-chrome
RUN mkdir -p /home/kasm-user/.config/GIMP/2.10 \
    && printf '# GIMP gimprc\n(icon-theme "Legacy")\n(icon-size medium)\n' \
       > /home/kasm-user/.config/GIMP/2.10/gimprc

# ===========================================================================
# 9. Disable GNOME Keyring password prompt
# ===========================================================================
RUN mv /usr/share/dbus-1/services/org.gnome.keyring.service /usr/share/dbus-1/services/org.gnome.keyring.service.disabled 2>/dev/null || true \
    && mv /usr/share/dbus-1/services/org.freedesktop.secrets.service /usr/share/dbus-1/services/org.freedesktop.secrets.service.disabled 2>/dev/null || true
RUN mkdir -p /home/kasm-user/.local/share/keyrings \
    && printf '[keyring]\ndisplay-name=Login\nctime=1234567890\nmtime=1234567890\nlock-on-idle=false\nlock-after=false\n' \
       > /home/kasm-user/.local/share/keyrings/login.keyring \
    && echo "login" > /home/kasm-user/.local/share/keyrings/default

# ===========================================================================
# 10. Custom GNOME session for KasmVNC
# ===========================================================================
COPY gnome-config/kasm.session /usr/share/gnome-session/sessions/kasm.session
RUN mkdir -p /etc/xdg/autostart.disabled \
    && for f in gnome-initial-setup-first-login.desktop \
                gnome-initial-setup-copy-worker.desktop \
                org.gnome.SettingsDaemon.UsbProtection.desktop \
                org.gnome.SettingsDaemon.Rfkill.desktop \
                org.gnome.SettingsDaemon.Wacom.desktop \
                org.gnome.SettingsDaemon.Wwan.desktop \
                org.gnome.SettingsDaemon.Smartcard.desktop \
                org.gnome.SettingsDaemon.Sharing.desktop \
                org.gnome.SettingsDaemon.PrintNotifications.desktop; do \
         [ -f /etc/xdg/autostart/$f ] && mv /etc/xdg/autostart/$f /etc/xdg/autostart.disabled/ || true; \
       done

# ===========================================================================
# 11. dconf defaults
# ===========================================================================
RUN mkdir -p /etc/dconf/profile /etc/dconf/db/local.d /etc/dconf/db/local.d/locks \
    && printf 'user-db:user\nsystem-db:local\n' > /etc/dconf/profile/user

RUN cat > /etc/dconf/db/local.d/00-nano-omni << 'DCONF_EOF'
[org/gnome/shell]
favorite-apps=['google-chrome.desktop', 'thunderbird.desktop', 'code.desktop', 'vlc.desktop', 'libreoffice-writer.desktop', 'libreoffice-calc.desktop', 'libreoffice-impress.desktop', 'gimp.desktop', 'org.gnome.Nautilus.desktop', 'org.gnome.Software.desktop', 'yelp.desktop']
enabled-extensions=['ubuntu-dock@ubuntu.com', 'ubuntu-appindicators@ubuntu.com']
welcome-dialog-last-shown-version='99.0'

[org/gnome/shell/extensions/dash-to-dock]
show-trash=true
show-mounts=false
show-show-apps-button=false

[org/gnome/desktop/background]
show-desktop-icons=false
picture-uri='file:///usr/share/backgrounds/warty-final-ubuntu.png'
picture-uri-dark='file:///usr/share/backgrounds/warty-final-ubuntu.png'

[org/gnome/shell/extensions/ding]
show-home=false
show-trash=false
show-volumes=false
show-network-volumes=false

[org/gnome/desktop/interface]
gtk-theme='Yaru'
icon-theme='Yaru'
cursor-theme='Yaru'
show-battery-percentage=false
toolkit-accessibility=false
enable-animations=false

[org/gnome/desktop/wm/preferences]
button-layout=':minimize,maximize,close'

[org/gnome/desktop/screensaver]
lock-enabled=false
idle-activation-enabled=false

[org/gnome/desktop/lockdown]
disable-lock-screen=true

[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/desktop/notifications]
show-in-lock-screen=false

[org/gnome/desktop/input-sources]
sources=[('xkb', 'us')]

[org/gnome/settings-daemon/plugins/power]
sleep-inactive-ac-type='nothing'
sleep-inactive-battery-type='nothing'

[org/gnome/mutter]
check-alive-timeout=uint32 0

[org/gnome/terminal/legacy]
theme-variant='dark'
DCONF_EOF

RUN cat > /etc/dconf/db/local.d/locks/00-nano-omni-locks << 'EOF'
/org/gnome/desktop/screensaver/lock-enabled
/org/gnome/desktop/screensaver/idle-activation-enabled
/org/gnome/desktop/lockdown/disable-lock-screen
/org/gnome/desktop/session/idle-delay
/org/gnome/settings-daemon/plugins/power/sleep-inactive-ac-type
/org/gnome/settings-daemon/plugins/power/sleep-inactive-battery-type
EOF

RUN dconf update

# ===========================================================================
# 12. D-Bus + polkit fixes
# ===========================================================================
RUN cat > /usr/share/dbus-1/system-services/org.freedesktop.timedate1.service << 'EOF'
[D-BUS Service]
Name=org.freedesktop.timedate1
Exec=/lib/systemd/systemd-timedated
User=root
SystemdService=dbus-org.freedesktop.timedate1.service
AssumedAppArmorLabel=unconfined
EOF
RUN mkdir -p /etc/polkit-1/localauthority/50-local.d \
    && cat > /etc/polkit-1/localauthority/50-local.d/10-timedate.pkla << 'EOF'
[Allow user timedate changes]
Identity=unix-user:kasm-user
Action=org.freedesktop.timedate1.set-time;org.freedesktop.timedate1.set-timezone;org.freedesktop.timedate1.set-local-rtc;org.freedesktop.timedate1.set-ntp
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF

# ===========================================================================
# 13. KasmVNC resolution + desktop_ready
# ===========================================================================
RUN sed -i 's/width: 1024/width: 1920/' /usr/share/kasmvnc/kasmvnc_defaults.yaml \
    && sed -i 's/height: 768/height: 1080/' /usr/share/kasmvnc/kasmvnc_defaults.yaml

RUN printf '#!/usr/bin/env bash\nuntil pids=$(pidof gnome-shell); do sleep .5; done\n' \
        > /usr/bin/desktop_ready \
    && chmod +x /usr/bin/desktop_ready

# ===========================================================================
# 14. Minimal desktop Flask API
# ===========================================================================
COPY desktop-api/main.py /home/kasm-user/server/main.py
RUN mkdir -p /home/kasm-user/server/screenshots

# ===========================================================================
# 15. Logind mock + KasmVNC entrypoint + custom GNOME startup
# ===========================================================================
COPY logind-mock.py /usr/local/bin/logind-mock.py
RUN chmod +x /usr/local/bin/logind-mock.py

COPY kasmvnc-entrypoint.sh /dockerstartup/kasmvnc-entrypoint.sh
RUN chmod +x /dockerstartup/kasmvnc-entrypoint.sh

COPY startup-scripts/custom_startup_gnome.sh /dockerstartup/custom_startup.sh
RUN chmod +x /dockerstartup/custom_startup.sh

# ===========================================================================
# 16. Fix ownership + environment
# ===========================================================================
RUN chown -R kasm-user:kasm-user /home/kasm-user

ENV LD_LIBRARY_PATH=
ENV VNC_RESOLUTION=1920x1080
ENV START_XFCE4=0
ENV START_DE=gnome-shell
ENV GNOME_SHELL_SESSION_MODE=ubuntu
ENV XDG_SESSION_TYPE=x11
ENV XDG_SESSION_DESKTOP=gnome
ENV XDG_CURRENT_DESKTOP=ubuntu:GNOME
ENV GDK_BACKEND=x11

ENV API_PORT=5000
ENV VNC_PORT=6901

EXPOSE 5000 6901
USER root
ENTRYPOINT ["/dockerstartup/kasmvnc-entrypoint.sh"]
CMD ["--wait"]
