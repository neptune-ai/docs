#!/bin/bash
# 
# Copyright (c) 2020, Neptune Labs
#

set -e

GENERATE_SETTINGS_SCRIPT="generate_settings.sh"

# Validation
if [ ! -d $WORKDIR ]; then
  echo "Directory $WORKDIR does not exit!"
  exit 1
fi

if [ ! -f $WORKDIR/$GENERATE_SETTINGS_SCRIPT ]; then
  echo "No settings generation script!"
  echo "$WORKDIR/$GENERATE_SETTINGS_SCRIPT"
  exit 2
fi

cd $WORKDIR

./$GENERATE_SETTINGS_SCRIPT

# Run application
nginx -g 'daemon off;'

