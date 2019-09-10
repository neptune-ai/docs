FROM nginx:stable

COPY docs/_build/html/ /usr/share/nginx/html

ADD docker/nginx.conf /etc/nginx/conf.d/default.conf

ADD docker/generate_settings.sh /generate_settings.sh
ADD docker/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
