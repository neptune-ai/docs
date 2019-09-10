#!/bin/bash
# 
# Copyright (c) 2019, Neptune Labs
#
# Runs neptune-frontend app.
#

set -e

GENERATE_SETTINGS_SCRIPT="generate_settings.sh"

# Validation
if [ ! -d $WORK_DIR ]; then
  echo "Directory $WORK_DIR does not exit!"
  exit 1
fi

if [ ! -f $WORK_DIR/$GENERATE_SETTINGS_SCRIPT ]; then
  echo "No settings generation script!"
  exit 2
fi

cd $WORK_DIR

/$GENERATE_SETTINGS_SCRIPT

# Run application
nginx -g 'daemon off;'

