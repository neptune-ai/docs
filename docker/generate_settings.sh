#! /bin/bash
# Copyright (c) 2020, Neptune Labs
#
# Sets various options from environemnt variables to docker-config.js
#

set -e

# Settings
WWWROOT="/usr/share/nginx/html"
CONFIG_FILE="$WWWROOT/docker-config.js"

echo "generating settings for docs from environment variables"

# Use default values if variables are not set
: ${IS_PRODUCTION:=false}

echo "
  window.dockerConfig = {
    isProduction: $IS_PRODUCTION
  };
" > $CONFIG_FILE

echo "See generated configuration:"
cat $CONFIG_FILE

