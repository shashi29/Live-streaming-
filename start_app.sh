#!/bin/bash

# Replace these variables with your actual values
APP_MODULE="app:app"
SERVICE_NAME="my_fastapi_app"

# Get the current working directory
APP_PATH=$(pwd)

# Create the systemd service unit file
cat <<EOF > "/etc/systemd/system/$SERVICE_NAME.service"
[Unit]
Description=My FastAPI Application
After=network.target

[Service]
ExecStartPre=/usr/bin/env bash -c 'YOUR_USERNAME=\$(whoami); YOUR_GROUPNAME=\$(id -gn); echo "User=\$YOUR_USERNAME\nGroup=\$YOUR_GROUPNAME" > /etc/systemd/system/$SERVICE_NAME.service.d/usergroup.conf'
ExecStart=/usr/bin/env python3 -m uvicorn $APP_MODULE --host 0.0.0.0 --port 8000
WorkingDirectory=$APP_PATH
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Create the usergroup.conf file for dynamic user and group
mkdir -p "/etc/systemd/system/$SERVICE_NAME.service.d/"
echo -e "YOUR_USERNAME=\$(whoami)\nYOUR_GROUPNAME=\$(id -gn)" > "/etc/systemd/system/$SERVICE_NAME.service.d/usergroup.conf"

# Reload systemd and start the service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME.service"
sudo systemctl start "$SERVICE_NAME.service"

# Check the status of the service
sudo systemctl status "$SERVICE_NAME.service"
